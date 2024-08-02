#!/usr/bin/env python
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
from torchsparse.utils.quantize import sparse_quantize

from mopa.common.solver.build import build_optimizer, build_scheduler
from mopa.common.utils.checkpoint import CheckpointerV2
from mopa.common.utils.logger import get_logger
from mopa.common.utils.metric_logger import MetricLogger, iou_to_excel
from mopa.common.utils.torch_util import set_random_seed
from mopa.common.utils.loss import BerhuLoss, l2_norm
from mopa.data.collate import inverse_to_all
from mopa.data.utils.visualize import debug_visualizer
from mopa.models.build import build_model_2d, build_model_3d
from mopa.data.build import build_dataloader
from mopa.data.utils.validate import validate
from mopa.models.losses import SupConLoss, entropy_loss
from mopa.data.mixmatch_ss import point_mixmatch, post_process


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/a2d2_semantic_kitti/xmuda.yaml',
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
        default='xMUDA',
        type=str,
        )
    parser.add_argument(
        '--resume_dir',
        help='Specific task name',
        # default='mopa/exp/models/0320_a2d2_semantic_kitti_xmuda_0',
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
    
    # contrastive criterion
    # if cfg.TRAIN.XMUDA.lambda_mm_ctr > 0:
    #     const_criterion = SupConLoss().cuda()

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
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
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
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        # copy data from cpu to gpu
        if 'SCN' in cfg.MODEL_3D.TYPE:
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
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
        
        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
            data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #
        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)

        # network-based postprocess
        if "SPVCNN" in cfg.MODEL_3D.TYPE:
            seg_logit_3d_cls = inverse_to_all(preds_3d['seg_logit'], data_batch_src)
            seg_logit_3d = inverse_to_all(preds_3d['seg_logit2'], data_batch_src)
        else:
            seg_logit_3d_cls = preds_3d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2']

        # segmentation loss: cross entropy
        loss_src_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        loss_src_3d = F.cross_entropy(seg_logit_3d_cls, data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(loss_src_2d=loss_src_2d, loss_src_3d=loss_src_3d)
        loss_2d = loss_src_2d
        loss_3d = loss_src_3d

        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d

        # depth prediction loss: BerhuLoss
        if cfg.TRAIN.DEPTH_PRED:
            depth_pred = preds_2d['depth_pred'].reshape(-1,1)
            depth_pred_loss = torch.sqrt(mse_loss(depth_pred, data_batch_src['depth_label']))
            # print(depth_pred_loss)
            train_metric_logger.update(depth_pred_loss=depth_pred_loss)
            loss_2d += cfg.TRAIN.DEPTH_PRED_COE.lambda_dp_src * depth_pred_loss
        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_src)
            train_metric_3d.update_dict(preds_3d, data_batch_src)

        # backward
        loss_2d.backward()
        loss_3d.backward()

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)

        # network-based postprocess
        if "SPVCNN" in cfg.MODEL_3D.TYPE:
            seg_logit_3d_cls = inverse_to_all(preds_3d['seg_logit'], data_batch_trg)
            seg_logit_3d = inverse_to_all(preds_3d['seg_logit2'], data_batch_trg)
        else:
            seg_logit_3d_cls = preds_3d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2']

        loss_2d = []
        loss_3d = []
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
                data_batch_trg['pseudo_label_2d'][mask_2d].long()
                )
            pl_loss_trg_3d = F.cross_entropy(
                seg_logit_3d_cls[mask_3d], 
                data_batch_trg['pseudo_label_3d'][mask_3d].long()
                )
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)

        if cfg.TRAIN.XMUDA.lambda_minent > 0:
            # MinEnt
            minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d['seg_logit'], dim=1))
            minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d['seg_logit'], dim=1))
            train_metric_logger.update(minent_loss_trg_2d=minent_loss_trg_2d,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_3d)

        sum(loss_2d).backward()
        sum(loss_3d).backward()

        optimizer_2d.step()
        optimizer_3d.step()

        # torch.cuda.empty_cache()
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
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

    for modality in ['2d', '3d']:
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
        abbrev_name='xMUDA'
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
