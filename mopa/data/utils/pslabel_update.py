import os
import numpy as np
import logging
import time
from tqdm import tqdm

import torch
import cv2
import torch.nn.functional as F
from mopa.data.pspath_sort import pspath_sort

from mopa.data.utils.evaluate import Evaluator
from mopa.data.utils.visualize import image_label_visualizer
from mopa.models.losses import entropy_loss, prob_2_entropy
from mopa.data.collate import inverse_to_all, range_to_point
from mopa.models.knn import KNN
from skimage.segmentation import slic, mark_boundaries

def cross_modal_lifting(preds_2d, img_indices):
    img_feats = []
    for i in range(preds_2d.shape[0]):
        img_feats.append(preds_2d[i][img_indices[i][:, 0], img_indices[i][:, 1]])
    img_feats = torch.cat(img_feats, 0)

    return img_feats

def pslabel_update(cfg,
             model_2d,
             model_3d,
             dataloader,
             pselab_dir=None,
             mix_match=False,
             entropy_fuse=False):

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_3d_all = Evaluator(class_names) if "RANGE" in cfg.DATASET_TARGET.TYPE \
                                                      else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None
    evaluator_ety = Evaluator(class_names) if entropy_fuse else None

    # initialize KNN
    if cfg.VAL.use_knn:
        post_knn = KNN(cfg.MODEL_3D.NUM_CLASSES)
        post_knn = post_knn.cuda()
    else:
        post_knn = None

    end = time.time()
    total_batches = len(dataloader)
    with torch.no_grad(), tqdm(total=total_batches, unit="%") as pbar:
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            data_batch['img'] = data_batch['img'].cuda()
            data_batch['seg_label'] = data_batch['seg_label'].cuda()
            # 3D input
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                if 'lidar' in data_batch.keys():
                    data_batch['lidar'] = data_batch['lidar'].cuda()
                else:
                    data_batch['x'][1] = data_batch['x'][1].cuda()
            elif 'RANGE' in cfg.DATASET_TARGET.TYPE:
                data_batch['proj_in'] = data_batch['proj_in'].float().cuda()
                data_batch['all_seg_label'] = data_batch['all_seg_label'].detach().numpy()
                if cfg.VAL.use_knn:
                    # * ONLY SUPPORT BATCH_SIZE = 1
                    assert cfg.VAL.BATCH_SIZE == 1
                    data_batch['proj_x'][0] = data_batch['proj_x'][0].cuda()
                    data_batch['proj_y'][0] = data_batch['proj_y'][0].cuda()
                    data_batch['proj_range'][0] = data_batch['proj_range'][0].cuda()
                    data_batch['unproj_range'][0] = data_batch['unproj_range'][0].cuda()
            else:
                raise NotImplementedError

            # predict
            if not mix_match:
                preds_2d = model_2d(data_batch)
            else:
                preds_2d = model_2d(data_batch['img'])
                preds_2d['seg_logit'] = cross_modal_lifting(preds_2d['seg_logit'], data_batch['img_indices'])
            preds_3d = model_3d(data_batch, save_feats=True) if model_3d else None
            
            # # DEBUG parts: Show 3D all ppredictions
            # image = data_batch['img'][0].cpu().numpy()
            # logit_2d_all = preds_2d['seg_logit_all'][0].cpu().numpy()
            # label_2d_all = np.argmax(logit_2d_all, axis=2)
            # image_label_visualizer(
            #     label_2d_all, image, "mopa/samples/all_pred_rgb.png")
            
            # mapping back to full label for SPVCNN
            if "SPVCNN" in cfg.MODEL_3D.TYPE:
                pred_logit_voxel_3d = inverse_to_all(preds_3d['seg_logit'], data_batch)
                pred_label_voxel_3d = pred_logit_voxel_3d.argmax(1).cpu().numpy()
            elif cfg.MODEL_3D.TYPE == "SalsaNext":
                pred_logit_3d_all, pred_logit_voxel_3d, \
                pred_label_3d_all, pred_label_voxel_3d = range_to_point(preds_3d['seg_logit'], 
                                                                        data_batch,
                                                                        post_knn=post_knn,
                                                                        post=cfg.VAL.use_knn,
                                                                        output_prob=cfg.VAL.knn_prob)
                pred_label_3d_all = pred_label_3d_all.cpu().numpy()
                pred_label_voxel_3d = pred_label_voxel_3d.cpu().numpy()
            else:
                pred_logit_voxel_3d = preds_3d['seg_logit']
                pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            
            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(pred_logit_voxel_3d, dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            val_2d_ety = prob_2_entropy(F.softmax(probs_2d, dim=1))
            val_3d_ety = prob_2_entropy(F.softmax(probs_3d, dim=1))
            if entropy_fuse:
                rv_ety_2d = 1 / (prob_2_entropy(probs_2d) + 1e-30)
                rv_ety_3d = 1 / (prob_2_entropy(probs_3d) + 1e-30)
                weight_2d = rv_ety_2d / (rv_ety_2d + rv_ety_3d)
                weight_3d = rv_ety_3d / (rv_ety_2d + rv_ety_3d)
                pred_label_ety_ensemble = (weight_2d * probs_2d + weight_3d * probs_3d).argmax(1).cpu().numpy()
                # print(pred_label_ensemble.shape, pred_label_ety_ensemble.shape)

            # get original point cloud from before voxelization
            # print(data_batch.keys())
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None
                pred_label_e_ensemble = pred_label_ety_ensemble[left_idx:right_idx] if entropy_fuse else None
                # print(pred_label_ensemble.shape, pred_label_ety_ensemble.shape)

                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)
                    if entropy_fuse:
                        evaluator_ety.update(pred_label_e_ensemble, curr_seg_label)


                if pselab_dir is not None:
                    assert np.all(pred_label_2d >= 0)
                    assert len(data_batch['lidar_path']) == 1   # only support batch 1
                    ps_label_pth = pspath_sort(cfg, data_batch['lidar_path'][0], pselab_dir)
                    curr_probs_2d = probs_2d[left_idx:right_idx]
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_dict = {
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy(),
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None,
                        'ori_keep_idx': data_batch['ori_keep_idx'][0],
                        'ori_img_points': data_batch['ori_img_points'][0],
                    }
                    np.save(ps_label_pth, pselab_data_dict, allow_pickle=True)

                left_idx = right_idx
            
            # update evaluator for all points
            if evaluator_3d_all:
                evaluator_3d_all.update(pred_label_3d_all, data_batch['all_seg_label'])
            end = time.time()
            
            # update progress bar
            progress = (iteration + 1) / total_batches * 100
            pbar.update(1)
            pbar.set_description(f"PsLabel Updating Progress: {progress:.2f}%")

        eval_list = [('2D', evaluator_2d)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
            if evaluator_3d_all:
                eval_list.extend([('3D_all', evaluator_3d_all)])
            if entropy_fuse:
                eval_list.extend([('entropy fuse', evaluator_ety)])
