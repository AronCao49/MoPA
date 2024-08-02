import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from typing import List
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


class BerhuLoss(nn.Module):
    """ Inverse Huber Loss """
    def __init__(self, ignore_index = 1):
        super(BerhuLoss, self).__init__()
        self.ignore_index = ignore_index
        self.l1 = torch.nn.L1Loss(reduction = 'none')

    def forward(self, prediction, ground_truth, imagemask=None):
        if imagemask is not None:
            mask = (ground_truth != self.ignore_index) & imagemask.to(torch.bool)
        else:
            mask = (ground_truth != self.ignore_index)
        difference = self.l1(torch.masked_select(prediction, mask), torch.masked_select(ground_truth, mask))
        with torch.no_grad():
            c = 0.2*torch.max(difference)
            mask = (difference <= c)

        lin = torch.masked_select(difference, mask)
        num_lin = lin.numel()

        non_lin = torch.masked_select(difference, ~mask)
        num_non_lin = non_lin.numel()

        total_loss_lin = torch.sum(lin)
        total_loss_non_lin = torch.sum((torch.pow(non_lin, 2) + (c**2))/(2*c))

        return (total_loss_lin + total_loss_non_lin)/(num_lin + num_non_lin)

def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1+1, batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # For feature MMD, until size(1)
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # For feature MMD, until size(1)
    # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))) # For corr MMD, until size(2)
    # total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))) # For corr MMD, until size(2)
    L2_distance = ((total0-total1)**2).sum(2) # For feature MMD, use only a single sum(2)
    # L2_distance = ((total0-total1)**2).sum(2).sum(2) # For corr MMD, use two sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

# Lovasz Softmax
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    if probas.dim() == 4:
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)
    

# focal loss
def focal_loss(preds: torch.Tensor, labels: torch.Tensor, 
               alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
    """
    Focal loss used in RatinaNet.

    Args:
        preds (Tensor): A float tensor of shape [N, K].
        labels (Tensor): a float tensor of label [N, ].
        alpha (float): Weighting factor in range (0, 1) to 
                        balance positive and negative examples.
        gamma (float): Exponent of modulating factor to balance easy vs hard samples.
        reduction (str): reduction method.
    Return: 
        Loss tensor
    """
    # reshape labels and one-hot encoding
    labels = labels.reshape(-1, 1)
    target = torch.nn.functional.one_hot(labels.long())

    # use the official focal loss function
    loss = torchvision.ops.focal_loss(preds, target, 
                                      alpha=alpha, gamma=gamma, reduction=reduction)
    
    # return
    return loss


def l2_norm(feats: torch.Tensor):
    """Simple L2 feature normalization"""
    
    norm = torch.linalg.norm(feats, ord=2, dim=1)
    epsilon = torch.Tensor([1e-8]).cuda()
    norm = torch.where(norm > epsilon, norm, torch.tensor(epsilon))
    normalized_feats = feats / norm.unsqueeze(1)
    
    return normalized_feats


def mask_cons_loss(
    all_logits: torch.Tensor,
    sam_mask_ls: List[torch.Tensor],
    min_entropy: bool = False
) -> torch.Tensor:
    """Function to compute intra-mask consistency lost

    Args:
        all_logits (torch.Tensor): all image logits (B, C, H, W).
        sam_mask_ls (List[torch.Tensor]): list of bool mask from SAM [[(H, W)]].
        min_entropy (bool, optional): whether to minimize mask mean logit.
    
    Returns:
        torch.Tensor: average consistency loss
    """
    all_img_loss = []
    num_classes = all_logits.shape[1]
    for batch_idx in range(len(sam_mask_ls)):
        batch_logits = all_logits[batch_idx, :, :, :]
        masks = sam_mask_ls[batch_idx]
        
        img_loss = []
        for mask_id in masks.unique():
            # pass invalid mask_id (-100 by default)
            if mask_id < 0:
                continue
            logits_in_mask = batch_logits[masks == mask_id]
            curr_loss = F.mse_loss(
                logits_in_mask, logits_in_mask.mean(dim=0, keepdim=True)
                )
            # minimize entropy of mask mean logits
            if min_entropy:
                curr_mean_logit = logits_in_mask.mean(dim=0)
                curr_loss -= torch.sum(curr_mean_logit * \
                    torch.log2(curr_mean_logit + 1e-30)) / np.log2(num_classes)
            img_loss.append(curr_loss)
        
        all_img_loss.append(sum(img_loss) / len(img_loss) if len(img_loss) != 0 else 0)
    
    if len(all_img_loss) != 0:
        return sum(all_img_loss) / len(all_img_loss)
    else:
        return 0
            

if __name__ == "__main__":
    test_logits = torch.softmax(torch.randn(1, 10, 302, 480).cuda(), dim=1)
    sam_mask_ls = [torch.randint(10, (302, 480)).cuda()]
    loss = mask_cons_loss(test_logits, sam_mask_ls, True)
    print(loss)