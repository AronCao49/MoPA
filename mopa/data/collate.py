import torch
import numpy as np
from functools import partial

from torch.nn.functional import normalize
import torch.nn.functional as F
import logging
from torchsparse.utils.collate import sparse_collate


def inverse_to_all(seg_logit, data_batch):
    # Testing line, will be removed after debugging
    num_point = 0
    indices_list = data_batch['indices']
    for inds in indices_list:
        num_point += inds.shape[0]
    # print(seg_logit.shape[0], num_point)
    # assert num_point == seg_logit.shape[0]
    # assert len(data_batch['inverse_map']) == len(indices_list)

    inv_seg_logit = []
    start_flag = 0
    # print(seg_logit.shape)
    for i in range(len(data_batch['inverse_map'])):
        map = data_batch['inverse_map'][i]
        end_flag = indices_list[i].shape[0]
        pred_label_3d = seg_logit[start_flag:start_flag+end_flag][map]
        inv_seg_logit.append(pred_label_3d)
        start_flag += end_flag
        # pred_label_voxel_3d = pred_label_voxel_3d[mask:]
    seg_logit_3d = torch.cat(inv_seg_logit, dim=0)
    return seg_logit_3d


def range_to_point(seg_logit, data_batch, post_knn=None, post=False, output_prob=False):
    """
    Function to project range image back to 3D points
    Args:
        seg_logit: range-style logit, torch tensor of shape (N, H, W, K).
        data_batch: python dictionary contains properties of range img.
        post_knn: KNN module to perform NN search
        post: bool to indicate whether KNN is used for postprocessing.
        output_prob: whether to compute prob average in NN
    Return:
        all_pc_logit: all 3D seg logit mapped from range-style logit
        sub_pc_logit: 3D seg logit mapped to image field
        all_pc_pred: all 3D pred mapped from range-style logit
        sub_pc_pred: 3D pred mapped to image field
    """
    keep_idx = data_batch['keep_idx']
    # loop over batch & check len of proj_x/y with seg_logit
    all_output_ls = []
    sub_output_ls = []
    assert seg_logit.shape[0] == len(data_batch['proj_x'])
    if post:
        if not post_knn:
            # * currently does not support post_proc without KNN
            logging.warning("Lack KNN to perform post-processing")
            raise AssertionError
        for i in range(seg_logit.shape[0]):
            sub_seg_logit = seg_logit[i] if output_prob else seg_logit[i].argmax(2)
            pc_output = post_knn(data_batch['proj_range'][i],
                                data_batch['unproj_range'][i],
                                sub_seg_logit,
                                data_batch['proj_x'][i],
                                data_batch['proj_y'][i],
                                output_prob=output_prob)
            sub_output_ls.append(pc_output[keep_idx[i]])
            all_output_ls.append(pc_output)
        # compute prob if not using output_prob
        if output_prob:
            all_pc_logit = torch.cat(all_output_ls, dim=0)
            sub_pc_logit = torch.cat(sub_output_ls, dim=0)
            all_pc_pred = all_pc_logit.argmax(1)
            sub_pc_pred = sub_pc_logit.argmax(1)
        else:
            all_pc_logit = []
            sub_pc_logit = []
            for i in range(seg_logit.shape[0]):
                sub_seg_logit = seg_logit[i]
                proj_x = data_batch['proj_x'][i]
                proj_y = data_batch['proj_y'][i]
                sub_pc_logit = sub_seg_logit[proj_y.long(), \
                                             proj_x.long(), :]
                sub_pc_logit.append(sub_pc_logit[keep_idx[i]])
                all_pc_logit.append(sub_pc_logit)
            all_pc_logit = torch.cat(all_pc_logit, dim=0)
            sub_pc_logit = torch.cat(sub_pc_logit, dim=0)
            all_pc_pred = torch.cat(all_output_ls, dim=0)
            sub_pc_pred = torch.cat(sub_output_ls, dim=0)
    else:
        for i in range(seg_logit.shape[0]):
            sub_seg_logit = seg_logit[i]
            proj_x = data_batch['proj_x'][i]
            proj_y = data_batch['proj_y'][i]
            sub_pc_logit = sub_seg_logit[proj_y.long(), \
                                         proj_x.long(), :]
            sub_output_ls.append(sub_pc_logit[keep_idx[i]])
            all_output_ls.append(sub_pc_logit)
        all_pc_logit = torch.cat(all_output_ls, dim=0)
        sub_pc_logit = torch.cat(sub_output_ls, dim=0)
        all_pc_pred = all_pc_logit.argmax(1)
        sub_pc_pred = sub_pc_logit.argmax(1)

    return all_pc_logit, sub_pc_logit, all_pc_pred, sub_pc_pred


def range_crop(proj_in):
    """
    Function to crop range img to value-existing area
    Args:
        proj_in: torch tensor of range img, of shape (N, H, W, 5)
    Return:
        crop_proj_in: cropped range img based on proj_in[:,:,:,0] > 0
    """
    h_index = torch.count_nonzero(proj_in[:,:,:,0], dim=1)
    h_min_idx = torch.min(h_index)
    h_max_idx = torch.max(h_index)
    w_index = torch.count_nonzero(proj_in[:,:,:,0], dim=2)
    w_min_idx = torch.min(w_index)
    w_max_idx = torch.max(w_index)

    return proj_in[:, h_min_idx:h_max_idx+1, w_min_idx:w_max_idx+1, :]

def collate_scn_base(
        input_dict_list, 
        output_orig, 
        output_image=True,
        output_ground=False,
        ):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_depth: whether to output depth points
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    labels=[]
    lidar_pth_ls = []
    scan_pth_ls = []
    aug_points_ls = []
    if output_image:
        imgs = []
        img_idxs = []
    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
        ori_keep_idx = []
        ori_img_points = []
    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    output_sam_mask = 'sam_mask' in input_dict_list[0].keys()
    output_ema_input = 'ori_img' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
        ori_pseudo_label_3d = []
    # for ground-based insertation
    if output_ground:
        ori_points_ls = []
        ori_feats_ls = []
        ori_obj_pc_ls = []
        ori_obj_label_ls = []
        ori_img_size_ls = []
        proj_matrix_ls = []
        g_indices_ls = []
    # for sam mask
    if output_sam_mask:
        sam_mask_ls = []
    # for ema input
    if output_ema_input:
        ori_img_ls = []
        ori_img_indices_ls = []
        ori_locs = []
        ori_feats = []
        ori_keep_idx_ls = []
        ori_idxs_ls = []

    for idx, input_dict in enumerate(input_dict_list):
        coords = torch.from_numpy(input_dict['coords'])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict['feats']))
        # ema input
        if output_ema_input:
            ori_img_ls.append(torch.from_numpy(input_dict['ori_img']))
            ori_img_indices_ls.append(input_dict['ori_img_indices'])
            ori_coords = torch.from_numpy(input_dict['ori_coords'])
            batch_idxs = torch.LongTensor(ori_coords.shape[0], 1).fill_(idx)
            ori_locs.append(torch.cat([ori_coords, batch_idxs], 1))
            ori_feats.append(torch.from_numpy(input_dict['ori_feats']))
            ori_keep_idx_ls.append(input_dict['aug_keep_idx'])
            ori_idxs_ls.append(input_dict['ori_idxs'])
        if 'aug_points' in input_dict.keys():
            aug_points_ls.append(input_dict['aug_points'])
        if 'scan_pth' in input_dict.keys():
            scan_pth_ls.append(input_dict['scan_pth']) 
        if 'lidar_path' in input_dict.keys():
            lidar_pth_ls.append(input_dict['lidar_path'])
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])
        if output_orig:
            if 'orig_seg_label' in input_dict.keys():
                orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
            if 'ori_keep_idx' in input_dict.keys():
                ori_keep_idx.append(input_dict['ori_keep_idx'])
                ori_img_points.append(input_dict['ori_img_points'])
        if output_pselab:
            ori_pseudo_label_3d.append(input_dict['ori_pseudo_label_3d'])
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
        # MoPA output
        if output_ground:
            ori_points_ls.append(input_dict['ori_points'])
            ori_feats_ls.append(input_dict['ori_feats'])
            ori_obj_pc_ls.append(input_dict['ori_obj_pc'])
            ori_obj_label_ls.append(input_dict['ori_obj_label'])
            ori_img_size_ls.append(input_dict['ori_img_size'])
            proj_matrix_ls.append(input_dict['proj_matrix'])
            if "g_indices" in input_dict.keys():
                g_indices_ls.append(input_dict['g_indices'])
        if output_sam_mask:
            sam_mask_ls.append(torch.from_numpy(input_dict['sam_mask']))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    out_dict['lidar_path'] = lidar_pth_ls
    out_dict['scan_pth_ls'] = scan_pth_ls
    # ema input
    if output_ema_input:
        ori_locs = torch.cat(ori_locs, 0)
        ori_feats = torch.cat(ori_feats, 0)
        out_dict['ori_x'] = [ori_locs, ori_feats]
        out_dict['ori_img'] = ori_img_ls
        out_dict['ori_img_indices'] = ori_img_indices_ls
        out_dict['ori_keep_idx'] = ori_keep_idx_ls
        out_dict['ori_idxs'] = ori_idxs_ls
    if len(aug_points_ls) > 0:
        out_dict['aug_points_ls'] = aug_points_ls
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
    if output_orig:
        if len(orig_seg_label) != 0:
            out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
        out_dict['ori_keep_idx'] = ori_keep_idx
        out_dict['ori_img_points'] = ori_img_points
    if output_pselab:
        out_dict['ori_pslabel_ls'] = ori_pseudo_label_3d
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    # MoPA output
    if output_ground:
        out_dict['ori_pc_ls'] = ori_points_ls
        out_dict['ori_feats_ls'] = ori_feats_ls
        out_dict['ori_obj_pc_ls'] = ori_obj_pc_ls
        out_dict['ori_obj_label_ls'] = ori_obj_label_ls
        out_dict['ori_img_size_ls'] = ori_img_size_ls
        out_dict['proj_mtx_ls'] = proj_matrix_ls
        if len(g_indices_ls) > 0:
            out_dict['g_indices_ls'] = g_indices_ls
    if output_sam_mask:
        out_dict['sam_mask_ls'] = sam_mask_ls

    return out_dict


def get_collate_scn(output_orig, output_ground):
    return partial(collate_scn_base,
                   output_orig=output_orig,
                   output_ground=output_ground)

def batch_mask_extractor(locs):
    batch_mask = []
    batch_tensor = locs[:, -1].int()
    # max_index = torch.max(batch_tensor).int()
    # min_index = torch.min(batch_tensor).int()
    # for idx in range(min_index, max_index):
    #     batch_mask.append(batch_tensor[batch_tensor == torch.LongTensor(idx)])
    batch_mask = torch.bincount(batch_tensor.int()).tolist()
    return batch_mask

