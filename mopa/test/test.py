#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import numpy as np
from yacs.config import CfgNode as CN

from mopa.common.utils.checkpoint import CheckpointerV2
from mopa.common.utils.logger import get_logger
from mopa.common.utils.metric_logger import MetricLogger, iou_to_excel
from mopa.common.utils.torch_util import set_random_seed
from mopa.models.build import build_model_2d, build_model_3d
from mopa.data.build import build_dataloader
from mopa.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/nuscenes/usa_singapore/xmuda_pl_pcmm_ema.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--task', default="MoPA", type=str, help='task name')
    parser.add_argument('--model_prefix',
                        #* Change ur model directory here
                        default='mopa/exp/models/MODEL_DIR_NAME',
                        type=str, help='path prefix to models dir')
    parser.add_argument('--ckpt2d',
                        default='best_val_xM_2d.pth', 
                        # default='model_2d_100000.pth',
                        type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('--ckpt3d',
                        default='best_val_xM_3d.pth',
                        # default='model_3d_100000.pth',
                        type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--pselab_dir', 
                        default=None,
                        type=str,  help='generate pseudo-labels')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test(cfg, logger, args, output_dir=''):
    # logger = logging.getLogger('xmuda.test')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    if args.ckpt2d:
        # load weight if specified
        weight_path = osp.join(args.model_prefix, args.ckpt2d)
        checkpointer_2d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_2d.load(None, resume=True)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    if args.ckpt3d:
        # load weight if specified
        weight_path = osp.join(args.model_prefix, args.ckpt3d)
        checkpointer_3d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_3d.load(None, resume=True)

    # build dataset
    test_dataloader = build_dataloader(cfg, mode='test', domain='target')

    if args.pselab_dir is not None:
        dataset_cfg = cfg.get('DATASET_TARGET')
        dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
        data_root_dir = dataset_kwargs['nuscenes_dir'] if 'nuscenes_dir' in dataset_kwargs.keys() \
            else dataset_kwargs['root_dir']
        pselab_dir = osp.join(data_root_dir, 'ps_label', args.pselab_dir)
        os.makedirs(pselab_dir, exist_ok=True)
        assert len(cfg.DATASET_TARGET.TEST) == 1
    else:
        pselab_dir = None

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    eval_dict = validate(
        cfg, 
        model_2d, 
        model_3d, 
        test_dataloader, 
        test_metric_logger, 
        logger, 
        pselab_dir=pselab_dir, 
    )
    
    if args.pselab_dir is None:
        iou_to_excel(eval_dict, osp.join(args.model_prefix, 'test_class_iou.xlsx'), eval_dict.keys())
        logger.info("Class-wise IoU saved to {}".format(osp.join(args.model_prefix, 'test_class_iou.xlsx')))


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
    # replace '@' with config path
    if output_dir:
        logs_output_dir = osp.join(args.model_prefix, 'logs')
        models_output_dir = osp.join(cfg.OUTPUT_DIR, 'models')
        config_path = osp.splitext(args.config_file)[0]

    # run name
    timestamp = time.strftime('%m%d')
    hostname = socket.gethostname()
    run_name = '{:s}-{:s}'.format(hostname, timestamp)

    log_file_pth = osp.join(
        logs_output_dir,
        "{}_test_{:s}_{}.log".format(args.task, run_name, os.environ['CUDA_VISIBLE_DEVICES'])
    )
    logger = get_logger(
        output=log_file_pth,
        abbrev_name='SS2MM'
        )
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    test(cfg, logger, args, models_output_dir)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    
    main()
