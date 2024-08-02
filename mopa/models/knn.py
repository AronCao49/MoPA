#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from doctest import OutputChecker
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size=5, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class KNN(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.knn = 5
        self.search = 5
        self.sigma = 1.0
        self.cutoff = 1.0
        self.nclasses = nclasses


    def forward(self, proj_range, unproj_range, proj_argmax, px, py, output_prob=False):
        ''' Warning! Only works for un-batched pointclouds.
            If they come batched we need to iterate over the batch dimension or do
            something REALLY smart to handle unaligned number of points in memory
        '''
        # get device
        if proj_range.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # sizes of projection scan
        H, W = proj_range.shape[0], proj_range.shape[1]

        # number of points
        P = unproj_range.shape

        # check if size of kernel is odd and complain
        if (self.search % 2 == 0):
            raise ValueError("Nearest neighbor kernel must be odd number")

        # calculate padding
        pad = int((self.search - 1) / 2)

        # unfold neighborhood to get nearest neighbors for each pixel (range image)
        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                      kernel_size=(self.search, self.search),
                                      padding=(pad, pad))

        # index with px, py to get ALL the pcld points
        idx_list = (py * W + px).long()
        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

        # WARNING, THIS IS A HACK
        # Make non valid (<0) range points extremely big so that there is no screwing
        # up the nn self.search
        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")

        # now the matrix is unfolded TOTALLY, replace the middle points with the actual range points
        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        # now compare range
        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

        # make a kernel to weigh the ranges according to distance in (x,y)
        # I make this 1 - kernel because I want distances that are close in (x,y)
        # to matter more
        inv_gauss_k = (
                1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

        # apply weighing
        k2_distances = k2_distances * inv_gauss_k

        # find nearest neighbors
        _, knn_idx = k2_distances.topk(
            self.knn, dim=1, largest=False, sorted=False)
        if output_prob:
            knn_argmax_out = self.knn_prob(knn_idx, 
                                        proj_argmax, 
                                        pad, 
                                        idx_list, 
                                        k2_distances,
                                        device,
                                        P)
        else:
            knn_argmax_out = self.knn_pred(knn_idx, 
                                        proj_argmax, 
                                        pad, 
                                        idx_list, 
                                        k2_distances,
                                        device,
                                        P)

        return knn_argmax_out

    def knn_pred(self, knn_idx, proj_argmax, pad, idx_list, k2_distances, device, P):
        # Two options to index knn candidate: (1) argmax ; (2) prob
        # do the same unfolding with the argmax
        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                        kernel_size=(self.search, self.search),
                                        padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]
        # get the top k predictions from the knn at each pixel
        knn_argmax = torch.gather(
                    input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        # fake an invalid argmax of classes + 1 for all cutoff items
        if self.cutoff > 0:
            knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
            knn_invalid_idx = knn_distances > self.cutoff
            if knn_argmax.dim() == 3:
                knn_argmax[knn_invalid_idx] = self.nclasses

        # now vote
        # argmax onehot has an extra class for objects after cutoff
        knn_argmax_onehot = torch.zeros(
            (1, self.nclasses + 1, P[0]), device=device).type(knn_argmax.type())
        ones = torch.ones_like(knn_argmax).type(knn_argmax.type())
        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

        # now vote (as a sum over the onehot shit)  (don't let it choose unlabeled OR invalid)
        knn_argmax_out = knn_argmax_onehot[:, 0:-1].argmax(dim=1)

        # reshape again
        knn_argmax_out = knn_argmax_out.view(P)
        
        return knn_argmax_out

    def knn_prob(self, knn_idx, proj_argmax, pad, idx_list, k2_distances, device, P):
        # Two options to index knn candidate: (2) prob
        # * additional prob-based knn search
        # do the same unfolding with the argmax

        # TODO: Bug found here, pending to fix
        proj_argmax = proj_argmax.permute(2, 0, 1)
        proj_unfold_1_argmax = F.unfold(proj_argmax[None, ...],
                                        kernel_size=(self.search, self.search),
                                        padding=(pad, pad)).to(device)
        proj_unfold_1_argmax = proj_unfold_1_argmax.view(1, self.nclasses, self.search**2, -1)
        proj_unfold_1_argmax = proj_unfold_1_argmax.permute(0, 2, 3, 1)
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list, :]
        
        # repeat knn_idx
        knn_idx = knn_idx.unsqueeze(-1).expand(knn_idx.shape[0],
                                               knn_idx.shape[1],
                                               knn_idx.shape[2], 
                                               self.nclasses)
        # get the top k predictions from the knn at each pixel
        knn_argmax = torch.gather(
                    input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        # fake an invalid argmax of classes + 1 for all cutoff items
        if self.cutoff > 0:
            knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx[:, :, :, 0])
            knn_invalid_idx = knn_distances > self.cutoff
            # knn_invalid_idx = knn_invalid_idx.unsqueeze(-1).repeat(knn_invalid_idx.shape[0],
            #                                                         knn_invalid_idx.shape[1],
            #                                                         knn_invalid_idx.shape[2], 
            #                                                         self.nclasses)
            knn_argmax[knn_invalid_idx] = 0

        # now vote
        # argmax onehot has an extra class for objects after cutoff
        knn_argmax_out = knn_argmax.sum(1)
        # knn_argmax_out = F.softmax(knn_argmax_out.float(), dim=2)
        knn_argmax_out = knn_argmax_out.view(P[0], self.nclasses)
        
        return knn_argmax_out


if __name__ == "__main__":
    import numpy as np
    from mopa.data.utils.augmentation_3d import range_projection

    post_knn = KNN(11).cuda()
    pc = np.fromfile('mopa/samples/test.bin')
    pc = pc.reshape((-1,4))
    data_dict = range_projection(pc, 0.392699, -0.392699, 2048, 64, False)
    
    # prepare input
    proj_range = torch.from_numpy(data_dict["proj_range"]).cuda()
    proj_x = torch.from_numpy(data_dict["proj_x"]).cuda()
    proj_y = torch.from_numpy(data_dict["proj_y"]).cuda()
    unproj_range = torch.from_numpy(data_dict["unproj_range"]).cuda()
    # Option 1: class voting
    # proj_argmax = torch.randint(0, 11, (64, 2048)).cuda()
    # Option 2: prob voting
    proj_argmax = torch.randn(64, 2048, 11)

    # test output
    print("Input point num: {}".format(unproj_range.shape))
    unproj_argamx = post_knn(proj_range, 
                             unproj_range, 
                             proj_argmax,
                             proj_x, 
                             proj_y,
                             output_prob=True)
    print("Output shape: {}".format(unproj_argamx.shape))
