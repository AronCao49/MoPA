"""xMUDA experiments configuration"""
import os.path as osp

from mopa.common.config.base import CN, _C

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'
# KNN search
_C.VAL.use_knn = False
_C.VAL.knn_prob = False

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLASS_WEIGHTS = []

# ---------------------------------------------------------------------------- #
# xMUDA options
# ---------------------------------------------------------------------------- #
_C.TRAIN.XMUDA = CN()
_C.TRAIN.XMUDA.lambda_xm_src = 0.0
_C.TRAIN.XMUDA.lambda_xm_trg = 0.0
_C.TRAIN.XMUDA.lambda_seq_trg = 0.0
_C.TRAIN.XMUDA.lambda_pl = 0.0
_C.TRAIN.XMUDA.lambda_minent = 0.0
_C.TRAIN.XMUDA.lambda_logcoral = 0.0
_C.TRAIN.XMUDA.knn_k = 5

# ---------------------------------------------------------------------------- #
# DA options
# ---------------------------------------------------------------------------- #
_C.TRAIN.DA_METHOD = None

# ---------------------------------------------------------------------------- #
# Depth Prediction options
# ---------------------------------------------------------------------------- #
_C.TRAIN.DEPTH_PRED_COE = CN()
_C.TRAIN.DEPTH_PRED_COE.lambda_dp_src = 0.0
_C.TRAIN.DEPTH_PRED_COE.lambda_dp_trg = 0.0

# ---------------------------------------------------------------------------- #
# Point Mix-Match
# ---------------------------------------------------------------------------- #
_C.TRAIN.PC_MM = CN()
_C.TRAIN.PC_MM.z_disc = None
# ground-based insertation arguments
_C.TRAIN.PC_MM.insert_mode = "ground"
_C.TRAIN.PC_MM.use_proj = True
_C.TRAIN.PC_MM.search_range = [25.0, 25.0]
_C.TRAIN.PC_MM.search_voxel_size = 0.5
_C.TRAIN.PC_MM.search_z_min = -2.0
_C.TRAIN.PC_MM.dis_range = ()
_C.TRAIN.PC_MM.use_class_weights = False
# post processing args
_C.TRAIN.PC_MM.proj_W = 1024
_C.TRAIN.PC_MM.proj_H = 64
_C.TRAIN.PC_MM.fov_up = 0.05235
_C.TRAIN.PC_MM.fov_down = -0.43633
_C.TRAIN.PC_MM.lambda_pc_mm = 0.0
_C.TRAIN.PC_MM.lambda_ctr_src = 0.0
_C.TRAIN.PC_MM.lambda_ctr_trg = 0.0
_C.TRAIN.PC_MM.lambda_sam_cons = 0.0
# Whether to use multi-stage ps labels
_C.TRAIN.PC_MM.ps_update_iter = 0
_C.TRAIN.PC_MM.ps_update_dir = None
# EMA args
_C.TRAIN.PC_MM.ema_start_iter = 100001
_C.TRAIN.PC_MM.ema_alpha_teacher = 0.999
_C.TRAIN.PC_MM.ema_update_period = 1
_C.TRAIN.PC_MM.ema_xm_prob = 0.7

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET_SOURCE = CN()
_C.DATASET_SOURCE.TYPE = ''
_C.DATASET_SOURCE.TRAIN = tuple()

_C.DATASET_TARGET = CN()
_C.DATASET_TARGET.TYPE = ''
_C.DATASET_TARGET.TRAIN = tuple()
_C.DATASET_TARGET.VAL = tuple()
_C.DATASET_TARGET.VAL_CORR = tuple()
_C.DATASET_TARGET.TEST = tuple()
_C.DATASET_TARGET.VISUAL = tuple()

# NuScenesSCN
_C.DATASET_SOURCE.NuScenesSCN = CN()
_C.DATASET_SOURCE.NuScenesSCN.preprocess_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.nuscenes_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.label_mode = 'object'
_C.DATASET_SOURCE.NuScenesSCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.NuScenesSCN.scale = 20
_C.DATASET_SOURCE.NuScenesSCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.NuScenesSCN.resize = (400, 225)
_C.DATASET_SOURCE.NuScenesSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation = CN()
_C.DATASET_SOURCE.NuScenesSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.NuScenesSCN.augmentation.flip_x = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.NuScenesSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.NuScenesSCN = CN(_C.DATASET_SOURCE.NuScenesSCN)
_C.DATASET_TARGET.NuScenesSCN.pselab_paths = tuple()
_C.DATASET_TARGET.NuScenesSCN.ps_label_dir = None
# MoPA args
_C.DATASET_TARGET.NuScenesSCN.g_indices_dir = None
_C.DATASET_TARGET.NuScenesSCN.use_pc_mm = False
_C.DATASET_TARGET.NuScenesSCN.multi_objs = False
_C.DATASET_TARGET.NuScenesSCN.obj_name_ls = []
_C.DATASET_TARGET.NuScenesSCN.obj_root_dir = None
_C.DATASET_TARGET.NuScenesSCN.z_disc = None
_C.DATASET_TARGET.NuScenesSCN.sc_rotation = None
_C.DATASET_TARGET.NuScenesSCN.use_sparse_quantize = False
# SAM consistency
_C.DATASET_TARGET.NuScenesSCN.sam_mask_dir = None
_C.DATASET_TARGET.NuScenesSCN.ema_input = False

# A2D2SCN
_C.DATASET_SOURCE.A2D2SCN = CN()
_C.DATASET_SOURCE.A2D2SCN.preprocess_dir = ''
_C.DATASET_SOURCE.A2D2SCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.A2D2SCN.scale = 20
_C.DATASET_SOURCE.A2D2SCN.full_scale = 4096
_C.DATASET_SOURCE.A2D2SCN.use_feats = False
_C.DATASET_SOURCE.A2D2SCN.use_sparse_quantize = False
# 2D
_C.DATASET_SOURCE.A2D2SCN.use_image = True
_C.DATASET_SOURCE.A2D2SCN.resize = (480, 302)
_C.DATASET_SOURCE.A2D2SCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation = CN()
_C.DATASET_SOURCE.A2D2SCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.A2D2SCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.A2D2SCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.color_jitter = (0.4, 0.4, 0.4)

# SemanticKITTISCN
_C.DATASET_SOURCE.SemanticKITTISCN = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.root_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.SemanticKITTISCN.scale = 20
_C.DATASET_SOURCE.SemanticKITTISCN.full_scale = 4096
_C.DATASET_SOURCE.SemanticKITTISCN.use_feats = False
_C.DATASET_SOURCE.SemanticKITTISCN.use_sparse_quantize = False
# 2D
_C.DATASET_SOURCE.SemanticKITTISCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.bottom_crop = (480, 302)
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.SemanticKITTISCN = CN(_C.DATASET_SOURCE.SemanticKITTISCN)
_C.DATASET_TARGET.SemanticKITTISCN.ps_label_dir = None
# MoPA
_C.DATASET_TARGET.SemanticKITTISCN.use_pc_mm = False
_C.DATASET_TARGET.SemanticKITTISCN.multi_objs = False
_C.DATASET_TARGET.SemanticKITTISCN.obj_name_ls = []
_C.DATASET_TARGET.SemanticKITTISCN.obj_root_dir = None
_C.DATASET_TARGET.SemanticKITTISCN.z_disc = None
_C.DATASET_TARGET.SemanticKITTISCN.sc_rotation = None
_C.DATASET_TARGET.SemanticKITTISCN.g_indices_dir = None
# SAM consistency
_C.DATASET_TARGET.SemanticKITTISCN.sam_mask_dir = None
_C.DATASET_TARGET.SemanticKITTISCN.ema_input = False

# ---------------------------------------------------------------------------- #
# Model 2D
# ---------------------------------------------------------------------------- #
_C.MODEL_2D = CN()
_C.MODEL_2D.TYPE = ''
_C.MODEL_2D.CKPT_PATH = ''
_C.MODEL_2D.NUM_CLASSES = 5
_C.MODEL_2D.DUAL_HEAD = False
_C.MODEL_2D.LOSS = "Default"
# ---------------------------------------------------------------------------- #
# UNetResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.UNetResNet34 = CN()
_C.MODEL_2D.UNetResNet34.pretrained = True
# ---------------------------------------------------------------------------- #
# DeepLabV3_ResNet50 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.DeepLabV3 = CN()
_C.MODEL_2D.DeepLabV3.pretrained = True

# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL_3D = CN()
_C.MODEL_3D.TYPE = ''
_C.MODEL_3D.CKPT_PATH = ''
_C.MODEL_3D.NUM_CLASSES = 5
_C.MODEL_3D.DUAL_HEAD = False
_C.MODEL_3D.LOSS = "Default"
# ----------------------------------------------------------------------------- #
# SCN options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SCN = CN()
_C.MODEL_3D.SCN.in_channels = 1
_C.MODEL_3D.SCN.m = 16  # number of unet features (multiplied in each layer)
_C.MODEL_3D.SCN.block_reps = 1  # block repetitions
_C.MODEL_3D.SCN.residual_blocks = False  # ResNet style basic blocks
_C.MODEL_3D.SCN.full_scale = 4096
_C.MODEL_3D.SCN.num_planes = 7
_C.MODEL_3D.SCN.pretrained = False
# ----------------------------------------------------------------------------- #
# SPVCNN options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SPVCNN = CN()
_C.MODEL_3D.SPVCNN.pretrained = False
_C.MODEL_3D.SPVCNN_Base = CN()
_C.MODEL_3D.SPVCNN_Base.pretrained = True
# ----------------------------------------------------------------------------- #
# SalsaNext options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SalsaNext = CN()
_C.MODEL_3D.SalsaNext.pretrained = False
_C.MODEL_3D.SalsaNext_Base = CN()
_C.MODEL_3D.SalsaNext_Base.pretrained = True
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('~/workspace/outputs/mopa/@')
