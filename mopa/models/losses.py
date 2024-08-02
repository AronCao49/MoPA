import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from mopa.models.xmuda_arch import batch_segment


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c = prob.size()
    ety = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    try:
        assert not torch.isnan(ety).any()
    except:
        raise AssertionError ("Nan found in ety: {}, input is {}".format(ety, prob))
    return ety

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x classes x points
        output: batch_size x 1 x points
    """
    # (num points, num classes)
    if v.dim() == 2:
        v = v.transpose(0, 1)
        v = v.unsqueeze(0)
    # (1, num_classes, num_points)
    assert v.dim() == 3
    n, c, p = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * p * np.log2(c))

def corr_distance(logit_2d_ls, logit_3d_ls):
    assert len(logit_2d_ls) == len(logit_3d_ls)
    corr_dis_ls = []
    for i in range(len(logit_2d_ls) // 2):
        # N1 x C, N2 x C

        logit_2d = logit_2d_ls[i]
        logit_3d = logit_3d_ls[i]



def logcoral_loss(x_src, x_trg):
    """
    Geodesic loss (log coral loss), reference:
    https://github.com/pmorerio/minimal-entropy-correlation-alignment/blob/master/svhn2mnist/model.py
    :param x_src: source features of size (N, ..., F), where N is the batch size and F is the feature size
    :param x_trg: target features of size (N, ..., F), where N is the batch size and F is the feature size
    :return: geodesic distance between the x_src and x_trg
    """
    # check if the feature size is the same, so that the covariance matrices will have the same dimensions
    assert x_src.shape[-1] == x_trg.shape[-1]
    assert x_src.dim() >= 2
    batch_size = x_src.shape[0]
    if x_src.dim() > 2:
        # reshape from (N1, N2, ..., NM, F) to (N1 * N2 * ... * NM, F)
        x_src = x_src.flatten(end_dim=-2)
        x_trg = x_trg.flatten(end_dim=-2)

    # subtract the mean over the batch
    x_src = x_src - torch.mean(x_src, 0)
    x_trg = x_trg - torch.mean(x_trg, 0)

    # compute covariance
    factor = 1. / (batch_size - 1)

    cov_src = factor * torch.mm(x_src.t(), x_src)
    cov_trg = factor * torch.mm(x_trg.t(), x_trg)

    # dirty workaround to prevent GPU memory error due to MAGMA (used in SVD)
    # this implementation achieves loss of zero without creating a fork in the computation graph
    # if there is a nan or big number in the cov matrix, use where (not if!) to set cov matrix to identity matrix
    condition = (cov_src > 1e30).any() or (cov_trg > 1e30).any() or torch.isnan(cov_src).any() or torch.isnan(cov_trg).any()
    cov_src = torch.where(torch.full_like(cov_src, condition, dtype=torch.uint8), torch.eye(cov_src.shape[0], device=cov_src.device), cov_src)
    cov_trg = torch.where(torch.full_like(cov_trg, condition, dtype=torch.uint8), torch.eye(cov_trg.shape[0], device=cov_trg.device), cov_trg)

    if condition:
        logger = logging.getLogger('xmuda.train')
        logger.info('Big number > 1e30 or nan in covariance matrix, return loss of 0 to prevent error in SVD decomposition.')

    _, e_src, v_src = cov_src.svd()
    _, e_trg, v_trg = cov_trg.svd()

    # nan can occur when taking log of a value near 0 (problem occurs if the cov matrix is of low rank)
    log_cov_src = torch.mm(v_src, torch.mm(torch.diag(torch.log(e_src)), v_src.t()))
    log_cov_trg = torch.mm(v_trg, torch.mm(torch.diag(torch.log(e_trg)), v_trg.t()))

    # Frobenius norm
    return torch.mean((log_cov_src - log_cov_trg) ** 2)

def CDAN(input_list, ad_net, dm_label, entropy=None, coeff=None, random_layer=None):
    # Conditional Domain ADV Loss from https://github.com/thuml/CDAN/blob/c904cd3f2fe092bb5175be1c5b6e42fa2036dece/pytorch/loss.py
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))   # op_out: [N, C, F]
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0)
    dc_target = (torch.ones(batch_size)*dm_label).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target.view(-1,1)) 


class SupConLoss(nn.Module):
    """
    Modified from Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, 
                labels_anchor: torch.Tensor,
                anchor_feature: torch.Tensor, 
                contrast_feature: torch.Tensor, 
                labels_contrast: torch.Tensor,
                iteration: int = None,
                ) -> torch.Tensor:
        """Function to compute class-wise contrastive loss

        Args:
            labels_anchor (torch.Tensor): the predicted label of anchor samples.
            anchor_feature (torch.Tensor): the corresponding anchor features.
            contrast_feature (torch.Tensor): the featue pool to be contrastive.
            labels_contrast (torch.Tensor): the label of featur in pool.

        Returns:
            torch.Tensor: class-wise contrastive loss.
        """

        # class-wise contrastive loss
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()          # N_a x N_q
        # tile mask
        anchor_mask = labels_anchor.reshape(-1,1).repeat(1, labels_contrast.shape[0])
        contrast_mask = labels_contrast.reshape(-1,1).repeat(1, labels_anchor.shape[0])
        mask = torch.eq(anchor_mask, contrast_mask.T).float().cuda()

        # compute log_prob
        exp_logits = (torch.exp(logits) + 1e-5) * (1 - mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)      # N_a / N_a
        # if not torch.all(mask.sum(1)):
        #     input("NaN found, pending debug...")

        # loss
        loss = -mean_log_prob_pos.mean()
        
        try:
            assert not torch.isnan(loss)
        except AssertionError:
            print("Current iteration: {}".format(iteration))
            raise AssertionError("NaN contrastive loss found")

        return loss


if __name__ == "__main__":
    a = torch.randn((3, 10))
    print("Input shape: {}".format(a.shape))

    entropy_a = prob_2_entropy(a)
    print("Entropy shape: {}".format(entropy_a.shape))


