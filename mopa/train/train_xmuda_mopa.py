#!/usr/bin/env python
from copy import deepcopy
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode as CN
from torch_ema import ExponentialMovingAverage

from mopa.common.solver.build import build_optimizer, build_scheduler
from mopa.common.utils.checkpoint import CheckpointerV2
from mopa.common.utils.logger import get_logger
from mopa.common.utils.metric_logger import MetricLogger, iou_to_excel
from mopa.common.utils.torch_util import set_random_seed
from mopa.data.collate import inverse_to_all
from mopa.data.utils.refine_pseudo_labels import refine_pseudo_labels
from mopa.models.build import build_model_2d, build_model_3d
from mopa.data.build import build_dataloader
from mopa.data.utils.validate import validate
from mopa.common.utils.loss import mask_cons_loss
from mopa.data.mixmatch_ss import point_mixmatch, post_process
from mopa.models.losses import prob_2_entropy

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/nuscenes/usa_singapore/xmuda_pl_pcmm_ema.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--task',
        help='Specific task name',
        default='MoPA',
        type=str,
        )
    parser.add_argument(
        '--resume_dir',
        help='Specific task name',
        default=None,
        type=str,
        )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def create_ema_model(model):
    ema_model = deepcopy(model)#get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    return ema_model


def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def train(cfg, logger, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(cfg.MODEL_2D.TYPE)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    logger.info('Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(cfg.MODEL_3D.TYPE)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    logger.info('Parameters: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build optimizer
        # build optimizer
    optim_2d_cfg = cfg.get('OPTIMIZER')['MODEL_2D']
    optim_3d_cfg = cfg.get('OPTIMIZER')['MODEL_3D']
    optimizer_2d = build_optimizer(optim_2d_cfg, model_2d)
    optimizer_3d = build_optimizer(optim_3d_cfg, model_3d)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None),
        'xM': None
    }
    best_metric_iter = {'2d': -1, '3d': -1, 'xM': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    if cfg.TRAIN.DEPTH_PRED:
        mse_loss = nn.MSELoss()
    for iteration in range(start_iteration, max_iteration):
        # create ema teacher if needed
        if iteration >= cfg.TRAIN.PC_MM.ema_start_iter and \
            ('ema_model_2d' not in locals() or 'ema_model_3d' not in locals()):
            ema_model_2d = ExponentialMovingAverage(
                model_2d.parameters(), decay=cfg.TRAIN.PC_MM.ema_alpha_teacher
            )
            ema_model_3d = ExponentialMovingAverage(
                model_3d.parameters(), decay=cfg.TRAIN.PC_MM.ema_alpha_teacher
            )
        
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        # copy data from cpu to gpu
        if 'SCN' in cfg.MODEL_3D.TYPE:
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            if iteration >= cfg.TRAIN.PC_MM.ema_start_iter:
                data_batch_trg['ori_x'][1] = data_batch_trg['ori_x'][1].cuda()
        elif 'SPVCNN' in cfg.MODEL_3D.TYPE:
            data_batch_src['lidar'] = data_batch_src['lidar'].cuda()
            data_batch_trg['lidar'] = data_batch_trg['lidar'].cuda()
        elif 'SalsaNext' in cfg.MODEL_3D.TYPE:
            data_batch_src['proj_in'] = data_batch_src['proj_in'].float().cuda()
            data_batch_trg['proj_in'] = data_batch_trg['proj_in'].float().cuda()
            data_batch_src['all_seg_label'] = data_batch_src['all_seg_label'].detach().numpy()
        else:
            raise NotImplementedError('The requested network {} is not supported for now.'.format(cfg.MODEL_3D.TYPE))
        # copy seg_label & image
        data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
        data_batch_src['img'] = data_batch_src['img'].cuda()
        data_batch_trg['img'] = data_batch_trg['img'].cuda()
        if iteration >= cfg.TRAIN.PC_MM.ema_start_iter:
            data_batch_trg['ori_img'] = [img.cuda() for img in data_batch_trg['ori_img']]
        # copy pseudo labels
        if cfg.TRAIN.XMUDA.lambda_pl > 0 and iteration < cfg.TRAIN.PC_MM.ema_start_iter:
            data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
            data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
        # copy sam masks
        if cfg.TRAIN.PC_MM.lambda_sam_cons > 0:
            data_batch_trg['sam_mask_ls'] = [mask.cuda() for mask in data_batch_trg['sam_mask_ls']]

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        
        # update pseudo labels with ema teacher
        if iteration >= cfg.TRAIN.PC_MM.ema_start_iter:
            with torch.no_grad():
                with ema_model_2d.average_parameters():
                    all_pred_2d = []
                    model_2d.eval()
                    for i in range(len(data_batch_trg['ori_img'])): 
                        all_pred_2d.append(model_2d({
                            'img': data_batch_trg['ori_img'][i].unsqueeze(0), 
                            'img_indices': [data_batch_trg['ori_img_indices'][i]]
                        })['seg_logit'])
                ema_seg_logit_2d = torch.cat(all_pred_2d, dim=0)
                
                with ema_model_3d.average_parameters():
                    model_3d.eval()
                    ema_seg_logit_3d = model_3d({
                        'x': data_batch_trg['ori_x']
                    })['seg_logit']
                probs_2d = F.softmax(ema_seg_logit_2d, dim=1)
                probs_3d = F.softmax(ema_seg_logit_3d, dim=1)
                
                # xM pslabel
                if np.random.uniform() <= cfg.TRAIN.PC_MM.ema_xm_prob:
                    # Option 1: ety ps-label fuse
                    rv_ety_2d = 1 / (prob_2_entropy(probs_2d) + 1e-30)
                    rv_ety_3d = 1 / (prob_2_entropy(probs_3d) + 1e-30)
                    weight_2d = rv_ety_2d / (rv_ety_2d + rv_ety_3d)
                    weight_3d = rv_ety_3d / (rv_ety_2d + rv_ety_3d)
                    probs_xm = (weight_2d * probs_2d + weight_3d * probs_3d)
                    
                    # refine output as pseudo labels
                    all_ps_label_2d = refine_pseudo_labels(
                        torch.max(probs_xm, dim=1)[0].cpu().numpy(), 
                        torch.argmax(probs_xm, dim=1).cpu().numpy()
                    )
                    all_ps_label_3d = refine_pseudo_labels(
                        torch.max(probs_xm, dim=1)[0].cpu().numpy(), 
                        torch.argmax(probs_xm, dim=1).cpu().numpy()
                    )
                else:
                    # Single modal pslabel
                    # refine output as pseudo labels
                    all_ps_label_2d = refine_pseudo_labels(
                        torch.max(probs_2d, dim=1)[0].cpu().numpy(), 
                        torch.argmax(probs_2d, dim=1).cpu().numpy()
                    )
                    all_ps_label_3d = refine_pseudo_labels(
                        torch.max(probs_3d, dim=1)[0].cpu().numpy(), 
                        torch.argmax(probs_3d, dim=1).cpu().numpy()
                    )
                
                # preserve only the ps_labels in FOV
                ps_label_2d = []
                ps_label_3d = []
                left_idx = 0
                for i in range(len(data_batch_trg['ori_keep_idx'])):
                    right_idx = left_idx + data_batch_trg['ori_keep_idx'][i].shape[0]
                    curr_ps_label_2d = all_ps_label_2d[left_idx:right_idx]
                    curr_ps_label_3d = all_ps_label_3d[left_idx:right_idx]
                    curr_keep_idx = data_batch_trg['ori_keep_idx'][i]
                    curr_idx = data_batch_trg['ori_idxs'][i]
                    ps_label_2d.append(curr_ps_label_2d[curr_keep_idx][curr_idx])
                    ps_label_3d.append(curr_ps_label_3d[curr_keep_idx][curr_idx])
                    left_idx = right_idx
                ps_label_2d = np.concatenate(ps_label_2d, axis=0)
                ps_label_3d = np.concatenate(ps_label_3d, axis=0)
                data_batch_trg.update({
                    'pseudo_label_2d': torch.from_numpy(ps_label_2d).cuda(),
                    'pseudo_label_3d': torch.from_numpy(ps_label_3d).cuda()
                })
                
                model_2d.train()
                model_3d.train()

        loss_2d = []
        loss_3d = []
        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #
        src_preds_2d = model_2d(data_batch_src)
        src_preds_3d = model_3d(data_batch_src)

        # network-based postprocess
        if "SPVCNN" in cfg.MODEL_3D.TYPE:
            src_seg_logit_3d_cls = inverse_to_all(src_preds_3d['seg_logit'], data_batch_src)
            src_seg_logit_3d = inverse_to_all(src_preds_3d['seg_logit2'], data_batch_src)
        else:
            src_seg_logit_3d_cls = src_preds_3d['seg_logit']
            src_seg_logit_3d = src_preds_3d['seg_logit2']

        # segmentation loss: cross entropy
        loss_src_2d = F.cross_entropy(
            src_preds_2d['seg_logit'], 
            data_batch_src['seg_label'], 
            weight=class_weights
            )
        loss_src_3d = F.cross_entropy(
            src_seg_logit_3d_cls, 
            data_batch_src['seg_label'], 
            weight=class_weights
            )
        train_metric_logger.update(
            loss_src_2d=loss_src_2d, loss_src_3d=loss_src_3d
            )
        
        # # ! Debug visualizer
        # img = data_batch_src['img'][0].detach().cpu().numpy()
        # img = (np.moveaxis(img, 0, -1) * 255.).astype(np.int32)
        # img_indices = data_batch_src['img_indices'][0]
        # seg_label = data_batch_src['seg_label'][0:img_indices.shape[0]]
        # seg_label = seg_label.detach().cpu().numpy()
        # draw_points_image_labels(
        #     img, 
        #     img_indices, 
        #     seg_label, 
        #     color_palette_type='SemanticKITTI', 
        #     point_size=3, save="mopa/samples/a2d2_sample.png")
        # input("Press input to continue")
        
        loss_2d.append(loss_src_2d)
        loss_3d.append(loss_src_3d)

        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            # cross-modal loss: KL divergence
            src_seg_logit_2d = src_preds_2d['seg_logit2'] \
                if cfg.MODEL_2D.DUAL_HEAD else src_preds_2d['seg_logit']
            xm_loss_src_2d = F.kl_div(
                F.log_softmax(src_seg_logit_2d, dim=1),
                F.softmax(src_preds_3d['seg_logit'].detach(), dim=1),
                reduction='none'
                ).sum(1).mean()
            xm_loss_src_3d = F.kl_div(
                F.log_softmax(src_seg_logit_3d, dim=1),
                F.softmax(src_preds_2d['seg_logit'].detach(), dim=1),
                reduction='none'
                ).sum(1).mean()
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d)

        # depth prediction loss: BerhuLoss
        if cfg.TRAIN.DEPTH_PRED:
            depth_pred = src_preds_2d['depth_pred'].reshape(-1,1)
            depth_pred_loss = torch.sqrt(mse_loss(depth_pred, data_batch_src['depth_label']))
            # print(depth_pred_loss)
            train_metric_logger.update(depth_pred_loss=depth_pred_loss)
            loss_2d.append(cfg.TRAIN.DEPTH_PRED_COE.lambda_dp_src * depth_pred_loss)
        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(src_preds_2d, data_batch_src)
            train_metric_3d.update_dict(src_preds_3d, data_batch_src)

        # backward
        sum(loss_2d).backward()
        sum(loss_3d).backward()

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #
        loss_2d = []
        loss_3d = []
        
        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)
                    
        # network-based postprocess
        if "SPVCNN" in cfg.MODEL_3D.TYPE:
            seg_logit_3d_cls = inverse_to_all(preds_3d['seg_logit'], data_batch_trg)
            seg_logit_3d = inverse_to_all(preds_3d['seg_logit2'], data_batch_trg)
        else:
            seg_logit_3d_cls = preds_3d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2']

        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d)
        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            # self-defined ignore mask
            ignore_label = -100
            mask_2d = data_batch_trg['pseudo_label_2d'] != ignore_label
            mask_3d = data_batch_trg['pseudo_label_3d'] != ignore_label
            # uni-modal self-training loss with pseudo labels
            pl_loss_trg_2d = F.cross_entropy(
                preds_2d['seg_logit'][mask_2d], 
                data_batch_trg['pseudo_label_2d'][mask_2d].long(),
                weight=class_weights if cfg.TRAIN.PC_MM.use_class_weights else None
                )
            pl_loss_trg_3d = F.cross_entropy(
                seg_logit_3d_cls[mask_3d], 
                data_batch_trg['pseudo_label_3d'][mask_3d].long(),
                weight=class_weights if cfg.TRAIN.PC_MM.use_class_weights else None
                )
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)

        # sam_mask consistency loss
        if cfg.TRAIN.PC_MM.lambda_sam_cons > 0:
            seg_logits_2d_all = F.softmax(preds_2d['seg_logit_all'], dim=3)
            sam_cons_loss = mask_cons_loss(
                seg_logits_2d_all,
                data_batch_trg['sam_mask_ls'],
                True
            )
            train_metric_logger.update(sam_cons_loss=sam_cons_loss)
            loss_2d.append(cfg.TRAIN.PC_MM.lambda_sam_cons * sam_cons_loss)
        
        # point mix-match
        if cfg.TRAIN.PC_MM.lambda_pc_mm > 0:
            dataset_cfg = cfg.get('DATASET_TARGET')
            dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
            augmentation = dataset_kwargs.pop('augmentation')
            pc_mm_kwargs = cfg.get('TRAIN').get('PC_MM')
            # Valid Ground-based Insertion
            if pc_mm_kwargs['insert_mode'] == "ground":
                start_time = time.time()
                ori_pc_ls = data_batch_trg['ori_pc_ls']
                ori_feats_ls = data_batch_trg['ori_feats_ls']
                ori_obj_pc_ls = data_batch_trg['ori_obj_pc_ls']
                ori_obj_label_ls = data_batch_trg['ori_obj_label_ls']
                proj_mtx_ls = data_batch_trg['proj_mtx_ls']
                ori_img_size_ls = data_batch_trg['ori_img_size_ls']
                front_axis = 'x' if "SemanticKITTI".upper() in dataset_cfg.TYPE.upper() \
                    else 'y'
                # Optional original pseudo labels
                # If not, then use all -100 as fake pseudo labels
                if 'ori_pslabel_ls' in data_batch_trg.keys():
                    ori_pslabel_ls = data_batch_trg['ori_pslabel_ls']
                else:
                    ori_pslabel_ls = [np.ones(ori_pc.shape[0]) * -100 \
                        for ori_pc in ori_pc_ls]
                
                # sample-wise overlap checkint & ground detection & insertation
                cat_pc_ls = []
                cat_pslabel_ls = []
                obj_mask_ls = []
                obj_ps_mask_ls = []
                for i in range(len(ori_pc_ls)):
                    ori_pc = np.concatenate(
                        (ori_pc_ls[i], ori_feats_ls[i].reshape(-1,1)), axis=1
                    )
                    cat_pc, cat_ps_label, obj_mask, obj_ps_mask = point_mixmatch(
                        ori_pc,
                        ori_pslabel_ls[i],
                        ori_obj_pc_ls[i],
                        ori_obj_label_ls[i],
                        proj_matrix=proj_mtx_ls[i],
                        image_size=ori_img_size_ls[i],
                        front_axis=front_axis,
                        g_indices=data_batch_trg['g_indices_ls'][i],
                        # grounding insert arugments
                        insert_mode=pc_mm_kwargs['insert_mode'],
                        search_voxel_size=pc_mm_kwargs['search_voxel_size'],
                        search_range=pc_mm_kwargs['search_range'],
                        search_z_min=pc_mm_kwargs['search_z_min'],
                    )
                    if obj_mask is not None:
                        # post processing if cat_pc exists
                        cat_pc_ls.append(cat_pc)
                        cat_pslabel_ls.append(cat_ps_label)
                        obj_mask_ls.append(obj_mask)
                        obj_ps_mask_ls.append(obj_ps_mask)
                        
                # post processing
                dataset_cfg = cfg.get("DATASET_TARGET")
                dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
                cat_input, cat_ps_label, obj_mask, _ = post_process(
                    cat_pc_ls, cat_pslabel_ls, obj_mask_ls,
                    dataset_kwargs['scale'], dataset_kwargs['full_scale'],
                    augmentation, scan_pth_ls=data_batch_trg['scan_pth_ls'],
                    use_proj=cfg.TRAIN.PC_MM.use_proj,
                    backbone=cfg.MODEL_3D.TYPE
                )
                if 'x' in cat_input.keys():
                    cat_input['x'][1] = cat_input['x'][1].cuda()
                if 'lidar' in cat_input.keys():
                    cat_input['lidar'] = cat_input['lidar'].cuda()
                cat_ps_label = cat_ps_label.cuda()
                obj_mask = obj_mask.cuda()
                g_insert_time = time.time() - start_time
                train_metric_logger.update(g_insert_time=g_insert_time)
                
                # forward
                cat_preds = model_3d(cat_input)
                ignore_label = -100
                cat_mask_3d = cat_ps_label != ignore_label
                cat_logit_3d = cat_preds['seg_logit']

                cat_loss_trg_3d = F.cross_entropy(
                    cat_logit_3d[cat_mask_3d], 
                    cat_ps_label[cat_mask_3d].long(),
                    weight=class_weights if cfg.TRAIN.PC_MM.use_class_weights else None
                )
                
            # record the obj segmentation performance
            obj_pred_label = torch.argmax(cat_logit_3d[obj_mask], dim=1)
            obj_gt_label = cat_ps_label[obj_mask]
            obj_acc = (obj_pred_label == obj_gt_label).int().sum() / \
                obj_gt_label.shape[0]
            train_metric_logger.update(pc_mm_loss=cat_loss_trg_3d)
            train_metric_logger.update(pc_mm_acc=obj_acc)
            loss_3d.append(cfg.TRAIN.PC_MM.lambda_pc_mm * cat_loss_trg_3d)

        sum(loss_2d).backward()
        sum(loss_3d).backward()

        optimizer_2d.step()
        optimizer_3d.step()
        scheduler_2d.step()
        scheduler_3d.step()

        # update ema_teacher
        if cfg.TRAIN.PC_MM.ema_update_period >= 0:
            if iteration >= cfg.TRAIN.PC_MM.ema_start_iter and \
                (iteration - cfg.TRAIN.PC_MM.ema_start_iter + 1) % cfg.TRAIN.PC_MM.ema_update_period == 0:
                ema_model_2d.update()
                ema_model_3d.update()
        
        torch.cuda.empty_cache()
        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou', 'ety')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            eval_dict = validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger,
                     logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d', 'xM']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if cur_iter >= (max_iteration / 2) and \
                        (best_metric[modality] is None or best_metric[modality] < cur_metric):
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter
                        # save best validation
                        if modality == '2d' or modality == 'xM':
                            checkpointer_2d.save('best_val_{}_2d'.format(modality), tag=False)
                        if modality == '3d' or modality == 'xM':
                            checkpointer_3d.save('best_val_{}_3d'.format(modality), tag=False)

            # restore training
            setup_train()

        end = time.time()

    for modality in ['2d', '3d', 'xM']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))
    
    iou_to_excel(eval_dict, osp.join(output_dir, 'val_class_iou.xlsx'), eval_dict.keys())
    logger.info("Class-wise IoU saved to {}".format(osp.join(output_dir, 'val_class_iou.xlsx')))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from mopa.common.config import purge_cfg
    from mopa.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    hostname = socket.gethostname()
    # replace '@' with config path
    # prepare checkpoint dir
    if args.resume_dir is None:
        if output_dir:
            models_output_dir = osp.join(cfg.OUTPUT_DIR, 'models')
            month_day = time.strftime('%m%d')
            spec_dir = osp.splitext(args.config_file)[0].replace('/', '_')
            spec_dir = month_day + spec_dir[7:] + '_' + os.environ['CUDA_VISIBLE_DEVICES']
            models_output_dir = osp.join(models_output_dir, spec_dir)
            flag = 1
            # check whether there exists a same dir. If so, generate a new one by adding number at the end
            while osp.isdir(models_output_dir):
                models_output_dir = models_output_dir + '-' + str(flag)
                flag += 1
                continue
            os.makedirs(models_output_dir, exist_ok=True)
    else:
        models_output_dir = args.resume_dir
    # prepare log dir
    logs_output_dir = osp.join(models_output_dir, 'logs')
    os.makedirs(logs_output_dir, exist_ok=True)
    

    # run name
    timestamp = time.strftime('%m%d')
    run_name = '{:s}-{:s}'.format(hostname, timestamp)

    log_file_pth = osp.join(
        logs_output_dir,
        "{}_train_{:s}_{}.log".format(args.task, run_name, os.environ['CUDA_VISIBLE_DEVICES'])
    )
    logger = get_logger(
        output=log_file_pth,
        abbrev_name='MoPA'
        )
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, logger, models_output_dir, run_name)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    import warnings
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    
    main()
