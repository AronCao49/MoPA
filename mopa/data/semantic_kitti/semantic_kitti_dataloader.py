import os.path as osp
import glob
from time import sleep
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from mopa.data.mixmatch_ss import point_mixmatch
from mopa.data.utils.refine_pseudo_labels import \
    refine_pseudo_labels, refine_sam_2Dlabels, refine_sam_mask
from mopa.data.utils.augmentation_3d import augment_and_scale_3d
from mopa.data.utils.visualize import debug_visualizer, image_label_visualizer
from mopa.data.semantic_kitti import splits

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SemanticKITTIBase(Dataset):
    """
    SemanticKITTI base dataset that store loading paths for lidar and images
    
    """

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    id_to_class_name = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    # use those categories if merge_classes == True (common with A2D2)
    categories = {
        'car': ['car', 'moving-car'],
        'truck': ['truck', 'moving-truck'],
        'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist',
                 'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
        'person': ['person', 'moving-person'],
        'road': ['road', 'lane-marking'],
        'parking': ['parking'],
        'sidewalk': ['sidewalk'],
        'building': ['building'],
        'nature': ['vegetation', 'trunk', 'terrain'],
        'other-objects': ['fence', 'traffic-sign', 'other-object', 'pole'],
    }

    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 ps_label_dir=None,
                 use_pc_mm=False,
                 obj_name_ls=[],
                 obj_root_dir=None,
                 g_indices_dir=None,
                 sam_mask_dir=None,
                 ):

        self.split = split
        # point mix-match
        self.obj_pc_dict = {}
        self.use_pc_mm = use_pc_mm
        self.obj_name_ls = obj_name_ls
        self.obj_root_dir = obj_root_dir
        self.g_indices_dir = g_indices_dir  # offline g_indices from PatchWork++
        self.sam_mask_dir = sam_mask_dir    # SAM
        scenes = []
        print("Initialize SemanticKITTI dataloader")
        assert isinstance(split, tuple)
        print('Load', split)

        # specify mapping table
        if merge_classes:
            self.categories = self.categories

        # retrieve scenes of specified split
        for single_split in self.split:
            scenes.extend(getattr(splits, single_split))
        self.root_dir = root_dir
        self.data = []

        # pseudo label dir
        self.ps_label_dir = ps_label_dir
        self.pselab_data = None
    
        # retrieve loading paths
        self.glob_frames(scenes)

        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def glob_frames(self, scenes):
        for scene in scenes:
            glob_path = osp.join(self.root_dir, 'dataset', 'sequences', scene, 'image_2', '*.png')
            cam_paths = sorted(glob.glob(glob_path))
            
            # load calibration
            calib = self.read_calib(osp.join(self.root_dir, 'dataset', 'sequences', scene, 'calib.txt'))
            proj_matrix = calib['P2'] @ calib['Tr']
            proj_matrix = proj_matrix.astype(np.float32)
            
            # load poses
            pose_f = open(osp.join(self.root_dir, 'dataset', 'sequences', scene, 'poses.txt'), 'r')
            poses = pose_f.readlines()
            pose_f.close()
            
            # pre-defined pseudo label pth if required
            ps_label_prefix = osp.join(self.root_dir, 'ps_label', self.ps_label_dir, scene) \
                if self.ps_label_dir is not None else None
            
            # prefix of g_indices from PatchWork++
            g_indices_prefix = osp.join(self.root_dir, self.g_indices_dir, scene) \
                if self.g_indices_dir is not None else None
            
            # SAM mask
            sam_mask_prefix = osp.join(self.root_dir, self.sam_mask_dir, scene) \
                if self.sam_mask_dir is not None else None

            for cam_path in cam_paths:
                basename = osp.basename(cam_path)
                frame_id = osp.splitext(basename)[0]
                assert frame_id.isdigit()

                # index pose from poses
                pose = poses[int(frame_id)].strip('\n').split(' ')
                pose_array = np.identity(4)
                pose_array[:3, :4] = np.asarray(pose).reshape(3,4)
                # transform pose from camera to lidar
                pose_array = np.linalg.inv(calib['Tr']) @ pose_array @ calib['Tr']

                data = {
                    'camera_path': cam_path,
                    'lidar_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'velodyne',
                                           frame_id + '.bin'),
                    'label_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'labels',
                                           frame_id + '.label'),
                    'proj_matrix': proj_matrix,
                    'pose': pose_array,
                    'scene': scene,
                    'frame_id': int(frame_id)
                }
                
                # update path to load pseudo label if required
                if ps_label_prefix is not None:
                    pslabel_path = osp.join(ps_label_prefix, frame_id + '.npy')
                    data['pslabel_path'] = pslabel_path

                # update path to load g_indices if required
                if g_indices_prefix is not None:
                    g_indices_path = osp.join(g_indices_prefix, frame_id + '.bin')
                    data['g_indices_pth'] = g_indices_path
                
                # update path to load sam mask
                if sam_mask_prefix is not None:
                    sam_mask_path = osp.join(sam_mask_prefix, frame_id + '.bin')
                    data['sam_mask_path'] = sam_mask_path
                
                for k, v in data.items():
                    if isinstance(v, str) and k != 'scene':
                        if not osp.exists(v):
                            raise IOError('File not found {}'.format(v))
                self.data.append(data)

        # load object pc
        if self.use_pc_mm:
            for obj_class in self.obj_name_ls:
                glob_path = osp.join(self.obj_root_dir, obj_class, "*.bin")
                obj_paths = sorted(glob.glob(glob_path))
                self.obj_pc_dict[obj_class] = obj_paths
    
    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
        return calib_out

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

class SemanticKITTISCN(SemanticKITTIBase):
    def __init__(self,
                 split,
                 root_dir,
                 ps_label_dir=None,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 image_normalizer=None,
                 # 3D augmentation
                 noisy_rot=0.0,  
                 flip_y=0.0,
                 rot_z=0.0,
                 transl=False,
                 # 2D augmentation (also effects 3D)
                 bottom_crop=tuple(),  
                 fliplr=0.0,
                 color_jitter=None,
                 output_orig=False,
                 backbone="SCN",
                 use_feats=False,
                 use_sparse_quantize=False,
                 # MoPA args
                 use_pc_mm=False,
                 multi_objs=False, 
                 g_indices_dir=None,
                 obj_name_ls=[],
                 obj_root_dir=None,
                 z_disc=None,
                 sc_rotation=None,
                 # SAM mask
                 sam_mask_dir=None,
                 # EMA input
                 ema_input=False
                 ):
        super().__init__(split,
                         root_dir,
                         merge_classes=merge_classes,
                         ps_label_dir=ps_label_dir,
                         use_pc_mm=use_pc_mm,
                         obj_name_ls=obj_name_ls,
                         obj_root_dir=obj_root_dir,
                         g_indices_dir=g_indices_dir,
                         sam_mask_dir=sam_mask_dir)

        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.image_normalizer = image_normalizer
        # 2D augmentation
        self.bottom_crop = bottom_crop
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

        # MoPA
        self.z_disc = z_disc
        self.sc_rotation = sc_rotation
        self.multi_objs = multi_objs

        # SPVCNN or not
        self.backbone = backbone
        self.use_feats = use_feats
        self.use_sparse_quantize = use_sparse_quantize

        # image mix-mixmatch
        self.mix_match = False
        
        # EMA input
        self.ema_input = ema_input


    def data_extraction(self, index):
        """
        Function to extract image and lidar from data_list based on index num
        Args:
            index: index number indicating the sample to be extracted
        Return;
            data_dict: dict containing related variables (e.g., point, image, label).
        """
        data_dict = self.data[index].copy()
        scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]
        feats = scan[:, 3]
        label = np.fromfile(data_dict['label_path'], dtype=np.uint32)
        label = label.reshape((-1))
        label = label & 0xFFFF  # get lower half for semantics

        z_idx = points[:, 2] > -3
        points = points[z_idx]
        feats = feats[z_idx]
        label = label[z_idx]

        # load pseudo label if needed
        if 'pslabel_path' in data_dict.keys():
            ps_data = np.load(data_dict['pslabel_path'], allow_pickle=True).tolist()

        # load g_indices if nedded
        if 'g_indices_pth' in data_dict.keys():
            g_indices = np.fromfile(data_dict['g_indices_pth'], dtype=np.int32)
            g_mask = np.zeros(scan.shape[0])
            g_mask[g_indices] = 1
            g_mask = g_mask[z_idx].astype(np.bool8)

        # load image
        image = Image.open(data_dict['camera_path'])
        
        data_dict['image'] = image
        data_dict['feats'] = feats
        data_dict['points'] = points
        data_dict['seg_labels'] = label.astype(np.int16)
        data_dict['all_seg_labels'] = label.astype(np.int16)
        data_dict['scan_pth'] = data_dict['lidar_path']
        # pseudo label
        if 'pslabel_path' in data_dict.keys():
            data_dict['pseudo_label_2d'] = ps_data['pseudo_label_2d']
            data_dict['pseudo_label_3d'] = ps_data['pseudo_label_3d']
            data_dict['probs_2d'] = ps_data['probs_2d']
            data_dict['probs_3d'] = ps_data['probs_3d']
            # to avoid non-determinstic
            data_dict['ori_keep_idx'] = ps_data['ori_keep_idx']
            data_dict['ori_img_points'] = ps_data['ori_img_points']
        
        # g_indices
        if 'g_indices_pth' in data_dict.keys():
            data_dict['g_indices'] = g_mask
        
        # load sam mask
        if 'sam_mask_path' in data_dict.keys():
            sam_mask = np.fromfile(data_dict['sam_mask_path'], dtype=np.uint8)
            data_dict['sam_mask'] = sam_mask.reshape(image.height, -1)  # (H, W)

        return data_dict
    
    
    def obj_sampling(self, obj_class):
        class_obj_ls = self.obj_pc_dict[obj_class]
        class_wise_id = np.random.choice(len(class_obj_ls), 1)
        obj_pc = np.fromfile(
            class_obj_ls[class_wise_id[0]], 
            dtype=np.float32).reshape((-1,4))
        try:
            assert not np.any(np.isnan(obj_pc))
        except:
            raise AssertionError("Found Nan object points: {}".format(
                class_obj_ls[class_wise_id[0]]
            ))
        obj_label = np.ones(obj_pc.shape[0]) * \
            self.label_mapping[self.class_name_to_id[obj_class]]
        
        return obj_pc, obj_label


    def preprocess(self, data_dict):
        points = data_dict['points']
        image_size = data_dict['image'].size

        # preserve the half of pc
        keep_idx = points[:, 0] > 0

        # ps_label refinement
        if 'pseudo_label_3d' in data_dict.keys():
            raw_ps_label_2d = data_dict['pseudo_label_2d'].astype(np.int32)
            ps_label_2d = refine_pseudo_labels(data_dict['probs_2d'],
                                               data_dict['pseudo_label_2d'].astype(np.int32))
            ps_label_3d = refine_pseudo_labels(data_dict['probs_3d'],
                                               data_dict['pseudo_label_3d'].astype(np.int32))
            data_dict.update({
                'pseudo_label_2d': ps_label_2d,
                'pseudo_label_3d': ps_label_3d,
            })
            keep_idx = data_dict['ori_keep_idx']
            img_points = data_dict['ori_img_points']
            
            if 'sam_mask' in data_dict.keys():
                # SAM refinement if exists
                probs_2d = np.zeros((data_dict['probs_2d'].shape[0], 10))
                probs_2d += np.expand_dims((1 - data_dict['probs_2d']) / 9, axis=1)
                
                probs_2d[np.arange(len(raw_ps_label_2d)),raw_ps_label_2d] = \
                    data_dict['probs_2d']
                full_2d_pslabels = refine_sam_2Dlabels(
                    probs_2d, img_points, data_dict['sam_mask'])
                
                sam_mask = refine_sam_mask(
                    data_dict['sam_mask'], 
                    max_h=image_size[1] - int(np.min(img_points, axis=0)[0]))
                
                data_dict.update({
                    'full_2d_pslabels': full_2d_pslabels,
                    'sam_mask': sam_mask
                })
            
            data_dict.update({
                'points': points[keep_idx],
                'feats': data_dict['feats'][keep_idx].reshape(-1,1),
                'seg_labels': data_dict['seg_labels'][keep_idx],
                'points_img': img_points,
                'ori_img_size': image_size,
                'ori_keep_idx': keep_idx,
            })
        else:
            points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
            img_points = np.matmul(data_dict['proj_matrix'].astype(np.float32), points_hcoords.T, dtype=np.float32).T

            img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
            img_points = np.around(img_points, decimals=2)
            keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image_size)
            keep_idx[keep_idx] = keep_idx_img_pts
            img_points = img_points[keep_idx_img_pts]
            # fliplr so that indexing is row, col and not col, row
            img_points = np.fliplr(img_points)
            # update current keep_idx & img_points
            data_dict.update({
                'ori_keep_idx': keep_idx,
                'ori_img_points': img_points
            })

            if 'sam_mask' in data_dict.keys():
                sam_mask = refine_sam_mask(
                    data_dict['sam_mask'], 
                    max_h=image_size[1] - int(np.min(img_points, axis=0)[0])
                )            
                data_dict.update({'sam_mask': sam_mask})
            
            data_dict.update({
                'points': points[keep_idx],
                'feats': data_dict['feats'][keep_idx].reshape(-1,1),
                'seg_labels': data_dict['seg_labels'][keep_idx],
                'points_img': img_points,
                'ori_img_size': image_size
            })

        # update g_indices to front view only
        if 'g_indices' in data_dict.keys():
            g_mask = data_dict['g_indices']
            data_dict.update({'g_indices': g_mask[keep_idx]})

        return data_dict
    

    def __getitem__(self, index):
        data_dict = self.data_extraction(index)
        all_seg_label = data_dict['all_seg_labels'].astype(np.int64)

        if self.use_pc_mm:
            # load object pc for MoPA
            obj_pc_ls = []
            obj_label_ls = []
            if not self.multi_objs:
                # Option 1: one object per scan
                obj_class = np.random.choice(len(self.obj_name_ls), 1)
                obj_class = self.obj_name_ls[obj_class[0]]
                obj_pc, obj_label = self.obj_sampling(obj_class)
                obj_pc_ls.append(obj_pc)
                obj_label_ls.append(obj_label)
            else:
                # Option 2: one object per class for each scan
                obj_pc_ls = []
                obj_label_ls = []
                for obj_class in self.obj_name_ls:
                    obj_pc, obj_label = self.obj_sampling(obj_class)
                    obj_pc_ls.append(obj_pc)
                    obj_label_ls.append(obj_label)
            data_dict.update({
                'ori_obj_pc': obj_pc_ls,
                'ori_obj_label': obj_label_ls,
                'full_scan': data_dict['points'],
                })
                    
        # preprocess
        data_dict = self.preprocess(data_dict)
        seg_label = data_dict['seg_labels']
        # mapping label to unifiend 
        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]
            all_seg_label = self.label_mapping[all_seg_label]

        out_dict = {}
        points = data_dict['points']
        feats = data_dict['feats']
        points_img = data_dict['points_img']
        image = data_dict['image'].copy()
        keep_idx = np.ones(len(points_img), dtype=np.bool_)
        
        # preserve unaugmented points and features
        ori_points = points.copy()
        ori_feats = feats.copy()
        # preserve unaugmented image
        if self.ema_input:
            ori_image = image.copy()
            ori_image = np.array(ori_image, dtype=np.float32, copy=False) / 255.
            ori_points_img = points_img.copy()
        
        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]

            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            # update point cloud
            points = points[keep_idx]
            feats = feats[keep_idx]
            seg_label = seg_label[keep_idx]
            
            # crop full pseudo labels
            if 'full_2d_pslabels' in data_dict.keys():
                data_dict['full_2d_pslabels'] = \
                    data_dict['full_2d_pslabels'][top:bottom, left:right]
            
            # crop sam mask
            # crop full pseudo labels
            if 'sam_mask' in data_dict.keys():
                data_dict['sam_mask'] = \
                    data_dict['sam_mask'][top:bottom, left:right]

        img_indices = points_img.astype(np.int64)

        # color jittering
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        
        # horizontal flip
        flip_idx = np.random.rand()
        if flip_idx < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]
            if 'full_2d_pslabels' in data_dict.keys():
                data_dict['full_2d_pslabels'] = \
                    np.ascontiguousarray(np.fliplr(data_dict['full_2d_pslabels']))
            if 'sam_mask' in data_dict.keys():
                data_dict['sam_mask'] = \
                    np.ascontiguousarray(np.fliplr(data_dict['sam_mask']))

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
        
        out_dict['img'] = np.moveaxis(image, -1, 0)
        out_dict['img_indices'] = img_indices
        if 'full_2d_pslabels' in data_dict.keys():
            out_dict['full_2d_pslabels'] = data_dict['full_2d_pslabels']
        if 'sam_mask' in data_dict.keys():
            out_dict['sam_mask'] = data_dict['sam_mask']

        # 3D data augmentation and scaling from points to voxel indices
        # Kitti lidar coordinates: x (front), y (left), z (up)
        coords, points = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_y=self.flip_y, rot_z=self.rot_z, transl=self.transl)
        coords = coords.astype(np.int64)
        
        # preserve unaugmented points & coords for EMA
        ori_coords, _ = augment_and_scale_3d(ori_points, self.scale, self.full_scale)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
        ori_idxs = (ori_coords.min(1) >= 0) * (ori_coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        if self.backbone.upper() == "SCN":
            out_dict['feats'] = np.ones([out_dict['coords'].shape[0], 1], np.float32)  # simply use 1 as feature
            if self.ema_input:
                out_dict['ori_img_indices'] = ori_points_img.astype(np.int64)
                out_dict['ori_img'] = np.moveaxis(ori_image, -1, 0)
                out_dict['ori_coords'] = ori_coords[ori_idxs]
                out_dict['ori_feats'] = np.ones([out_dict['ori_coords'].shape[0], 1], np.float32)
                out_dict['aug_keep_idx'] = keep_idx
                out_dict['ori_idxs'] = idxs
            
        out_dict['seg_label'] = seg_label[idxs]
        out_dict['img_indices'] = out_dict['img_indices'][idxs]
        out_dict['lidar_path'] = data_dict['lidar_path']
        out_dict['scan_pth'] = data_dict['scan_pth']

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'orig_points_idx': idxs,
                'ori_keep_idx': data_dict['ori_keep_idx'],          # for non-determinstic
                'ori_img_points': data_dict['ori_img_points']# for non-determinstic
            })

        if 'pseudo_label_2d' in data_dict.keys():
            ps_label_2d = data_dict['pseudo_label_2d']
            ps_label_3d = data_dict['pseudo_label_3d']
            out_dict.update({
                'pseudo_label_2d': ps_label_2d[keep_idx][idxs],
                'pseudo_label_3d': ps_label_3d[keep_idx][idxs],
                'ori_pseudo_label_3d': ps_label_3d
            })

        if self.use_pc_mm:
            out_dict.update({
                'ori_obj_pc': data_dict['ori_obj_pc'],
                'ori_obj_label': data_dict['ori_obj_label'],
                'ori_img_size': data_dict['ori_img_size'],
                'proj_matrix': data_dict['proj_matrix'],
                'ori_points': ori_points,
                'ori_feats': ori_feats
            })
            if 'g_indices' in data_dict.keys():
                g_mask = data_dict['g_indices']
                out_dict.update({'g_indices': g_mask})

        return out_dict


def test_SemanticKITTISCN():
    from mopa.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    from tqdm import tqdm
    np.random.seed(42)
    root_dir = 'mopa/datasets/semantic_kitti'
    ps_label_dir = '1226_ps_label'
    # ps_label_dir = None
    split = ('train',)
    # split = ('val',)
    use_pc_mm = True
    obj_name_ls = ['truck']
    obj_root_dir = 'mopa/datasets/waymo/waymo_extracted/objects'
    dataset = SemanticKITTISCN(split=split,
                               root_dir=root_dir,
                               ps_label_dir=ps_label_dir,
                               merge_classes=True,
                               output_orig=True,
                               use_pc_mm=use_pc_mm,
                               obj_name_ls=obj_name_ls,
                               obj_root_dir=obj_root_dir,
                            #    bottom_crop=(480, 302),
                               g_indices_dir="g_indices",
                               sam_mask_dir="img_mask",
                               ema_input=True
                               )
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        import matplotlib.pyplot as plt
        from mopa.data.utils.visualize import debug_visualizer, \
            A2D2_COLOR_PALETTE_SHORT, show_anns, draw_point_image_depth
        ref_pc = data['coords'] / 20
        ref_pc = ref_pc.astype(np.float32)
        ori_q_pc = data['ori_coords'] / 20
        ori_q_pc = ori_q_pc.astype(np.float32)
        assert ref_pc.shape[0] == ori_q_pc.shape[0]

        # Check sam_mask
        img = data['img'] * 255
        img = np.transpose(img.astype(np.uint8), axes=[1,2,0])
        sam_mask = data['sam_mask']
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        show_anns(sam_mask, img)
        plt.axis('off')
        plt.savefig("mopa/samples/sam_sample.jpg")
        a = debug_visualizer(data['ori_points'], 'mopa/samples/00_00000_raw.pcd')
        
        # Offline g_indices visualization
        g_indices = data['g_indices']
        ori_points = data['ori_points']
        assert g_indices.shape[0] == ori_points.shape[0]
        colors = np.zeros([ori_points.shape[0], 3])
        colors[g_indices, 0] = 1

        a = debug_visualizer(ori_points, "mopa/samples/offline_g_indices.pcd", colors)
        input("Press Enter to Continue...")
    
    print("Completed looping semantic kitti, nothing wrong!")

def compute_class_weights():
    preprocess_dir = 'mopa/datasets/semantic_kitti/preprocess/preprocess'
    split = ('train',)
    dataset = SemanticKITTIBase(split,
                                preprocess_dir,
                                merge_classes=True
                                )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        labels = dataset.label_mapping[data['seg_labels']]
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    test_SemanticKITTISCN()
    # compute_class_weights()
