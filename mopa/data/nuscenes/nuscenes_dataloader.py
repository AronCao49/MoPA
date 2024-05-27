import os.path as osp
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.ndimage import zoom

from mopa.data.utils.refine_pseudo_labels import refine_pseudo_labels, refine_sam_mask
from mopa.data.utils.augmentation_3d import augment_and_scale_3d


class NuScenesBase(Dataset):
    """NuScenes dataset"""

    class_names_seg = [
        'ignore',
        'barrier',
        'bicycle',
        'bus',
        'car',
        'construction_vehicle',
        'motorcycle',
        'pedestrian',
        'traffic_cone',
        'trailer',
        'truck',
        'driveable_surface',
        'other_flat',
        'sidewalk',
        'terrain',
        'manmade',
        'vegetation'
     ]

    # use those categories if merge_classes == True
    categories_seg = {
        "vehicle": ["bicycle", "bus", "car", "construction_vehicle", "motorcycle", "trailer", "truck"],
        "driveable_surface": ["driveable_surface"],
        "sidewalk": ["sidewalk"],
        "terrain": ["terrain"],
        "manmade": ["manmade"],
        "vegetation": ["vegetation"],
        # "ignore": ["ignore", "barrier", "pedestrian", "traffic_cone", "other_flat"],
    }
    
    # Class map of NuScenes object detection
    class_names_obj = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
        "background",
    ]
    # use those categories if merge_classes == True
    categories_obj = {
        "vehicle": ["car", "truck", "bus", "trailer", "construction_vehicle"],
        "pedestrian": ["pedestrian"],
        "bike": ["motorcycle", "bicycle"],
        "traffic_boundary": ["traffic_cone", "barrier"],
        "background": ["background"]
    }
    
    

    def __init__(self,
                 split,
                 preprocess_dir,
                 label_mode, 
                 merge_classes=False,
                 pselab_paths=None
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize Nuscenes dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        self.pselab_data = None
        if pselab_paths:
            assert isinstance(pselab_paths, tuple)
            print('Load pseudo label data ', pselab_paths)
            self.pselab_data = []
            for curr_split in pselab_paths:
                self.pselab_data.extend(np.load(curr_split, allow_pickle=True))

            # check consistency of data and pseudo labels
            assert len(self.pselab_data) == len(self.data)
            for i in range(len(self.pselab_data)):
                assert len(self.pselab_data[i]['pseudo_label_2d']) == len(self.data[i]['seg_labels'])

            # refine 2d pseudo labels
            probs2d = np.concatenate([data['probs_2d'] for data in self.pselab_data])
            pseudo_label_2d = np.concatenate([data['pseudo_label_2d'] for data in self.pselab_data]).astype(np.int)
            pseudo_label_2d = refine_pseudo_labels(probs2d, pseudo_label_2d)

            # refine 3d pseudo labels
            # fusion model has only one final prediction saved in probs_2d
            if self.pselab_data[0]['probs_3d'] is not None:
                probs3d = np.concatenate([data['probs_3d'] for data in self.pselab_data])
                pseudo_label_3d = np.concatenate([data['pseudo_label_3d'] for data in self.pselab_data]).astype(np.int)
                pseudo_label_3d = refine_pseudo_labels(probs3d, pseudo_label_3d)
            else:
                pseudo_label_3d = None

            # undo concat
            left_idx = 0
            for data_idx in range(len(self.pselab_data)):
                right_idx = left_idx + len(self.pselab_data[data_idx]['probs_2d'])
                self.pselab_data[data_idx]['pseudo_label_2d'] = pseudo_label_2d[left_idx:right_idx]
                if pseudo_label_3d is not None:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = pseudo_label_3d[left_idx:right_idx]
                else:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = None
                left_idx = right_idx

        if merge_classes:
            if label_mode == "object":
                self.class_names = self.class_names_obj
                self.categories = self.categories_obj
                self.ori_class_names = self.class_names_obj
            else:
                self.class_names = self.class_names_seg
                self.categories = self.categories_seg
                self.ori_class_names = self.class_names_seg
            
            self.label_mapping = -100 * np.ones(len(self.class_names), dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
            
        else:
            self.ori_class_names = self.class_names_obj if label_mode == "object" \
                else self.class_names_seg
            self.ori_class_names = self.ori_class_names[1:] \
                if 'ignore' in self.ori_class_names else self.ori_class_names
            self.label_mapping = np.arange(-1, len(self.ori_class_names))
            self.label_mapping[0] = -100    # ignore class as -100
            self.class_names = self.ori_class_names

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class NuScenesSCN(NuScenesBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 label_mode,
                 nuscenes_dir='',
                 pselab_paths=None,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 use_sparse_quantize=False,
                 resize=(400, 225),
                 image_normalizer=None,
                 # 3D augmentation
                 noisy_rot=0.0,
                 flip_x=0.0,
                 rot_z=0.0,
                 transl=False,
                 # 2D augmentation
                 fliplr=0.0,
                 color_jitter=None,
                 output_orig=False,
                 ps_label_dir=None,
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
                 ema_input=False,
                 ):
        super().__init__(split,
                         preprocess_dir,
                         label_mode,
                         merge_classes=merge_classes,
                         pselab_paths=pselab_paths)

        self.nuscenes_dir = nuscenes_dir
        self.output_orig = output_orig
        self.ps_label_dir = ps_label_dir

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.resize = resize
        self.image_normalizer = image_normalizer

        # data augmentation
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        
        # MoPA
        self.use_pc_mm = use_pc_mm
        self.multi_objs = multi_objs
        self.obj_name_ls = obj_name_ls
        self.obj_root_dir = obj_root_dir
        self.g_indices_dir = g_indices_dir
        
        # SAM mask
        self.sam_mask_dir = sam_mask_dir
        
        # load objects from obj_root_dir
        if self.use_pc_mm:
            self.obj_pc_dict = {}
            for obj_class in self.obj_name_ls:
                glob_path = osp.join(self.obj_root_dir, obj_class, "*.bin")
                obj_paths = sorted(glob.glob(glob_path))
                self.obj_pc_dict[obj_class] = obj_paths
        
        # EMA input
        self.ema_input = ema_input

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
        # Simple fix for class name mismatch
        obj_class = 'pedestrian' if obj_class == 'person' else obj_class
        obj_label = np.ones(obj_pc.shape[0]) * \
            self.label_mapping[self.ori_class_names.index(obj_class)]
        
        return obj_pc, obj_label
    
    
    def __getitem__(self, index):
        data_dict = self.data[index]
        
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

        points = data_dict['points'].copy()
        seg_label = data_dict['seg_labels'].astype(np.int64)
        lidar_path = data_dict['lidar_path']
        lidar_prefix, lidar_file = \
            lidar_path.split('/')[-2], lidar_path.split('/')[-1]
        camera_prefix, camera_file = \
            data_dict['camera_path'].split('/')[-2], data_dict['camera_path'].split('/')[-1]
        
        # load g_indices if needed
        #! Pending fix: the loaded g_indices might be of full scan
        if self.g_indices_dir is not None:
            g_indices_prefix = osp.join(
                self.nuscenes_dir, self.g_indices_dir, lidar_prefix
            )
            g_indices_path = osp.join(g_indices_prefix, lidar_file)
            g_indices = np.fromfile(g_indices_path, dtype=np.int32)
            g_mask = np.zeros(data_dict['valid_mask'].shape[0])
            g_mask[g_indices] = 1
            g_mask = g_mask[data_dict['valid_mask']]
            data_dict['g_indices'] = g_mask.astype(np.bool8)

        # load SAM mask if needed
        if self.sam_mask_dir is not None:
            assert len(self.split) == 1
            sam_mask_prefix = osp.join(
                self.nuscenes_dir, self.sam_mask_dir, self.split[0], camera_prefix
            )
            sam_mask_path = osp.join(sam_mask_prefix, camera_file.replace('.jpg', '.bin'))
            data_dict['sam_mask'] = np.fromfile(sam_mask_path, dtype=np.uint8)
            
        # load pseudo labels if needed
        if self.ps_label_dir is not None:
            ps_label_prefix = osp.join(
                self.nuscenes_dir, self.ps_label_dir, lidar_prefix
            )
            ps_label_path = osp.join(ps_label_prefix, lidar_file.replace('.bin', '.npy'))
            ps_data = np.load(ps_label_path, allow_pickle=True).tolist()
            data_dict['pseudo_label_2d'] = ps_data['pseudo_label_2d']
            data_dict['pseudo_label_3d'] = ps_data['pseudo_label_3d']
            data_dict['probs_2d'] = ps_data['probs_2d']
            data_dict['probs_3d'] = ps_data['probs_3d']
        
        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        out_dict = {}

        points_img = data_dict['points_img'].copy()
        img_path = osp.join(self.nuscenes_dir, data_dict['camera_path'])
        image = Image.open(img_path)
        data_dict.update({"ori_img_size": image.size})
        ori_image_H = image.height
            
        if self.resize:
            if not image.size == self.resize:
                # check if we do not enlarge downsized images
                assert image.size[0] > self.resize[0]

                # scale image points
                points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
                points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])

                # resize image
                image = image.resize(self.resize, Image.BILINEAR)

                if 'sam_mask' in data_dict.keys():
                    sam_mask = data_dict['sam_mask'].reshape(ori_image_H, -1)
                    scale_f_H = self.resize[0] / data_dict['ori_img_size'][0]
                    scale_f_W = self.resize[1] / data_dict['ori_img_size'][1]
                    sam_mask = zoom(
                        sam_mask, (scale_f_H, scale_f_W), order=0
                    )
                    sam_mask = refine_sam_mask(
                        sam_mask, 
                        max_h=image.size[1] - int(np.min(points_img, axis=0)[0])
                    )            
                    data_dict.update({'sam_mask': sam_mask})
                    
        if self.ema_input:
            ori_image = image.copy()
            ori_image = np.array(ori_image, dtype=np.float32, copy=False) / 255.
            ori_points_img = points_img.copy()

        img_indices = points_img.astype(np.int64)

        assert np.all(img_indices[:, 0] >= 0)
        assert np.all(img_indices[:, 1] >= 0)
        assert np.all(img_indices[:, 0] < image.size[1])
        assert np.all(img_indices[:, 1] < image.size[0])

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]
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

        # 3D data augmentation and scaling from points to voxel indices
        # nuscenes lidar coordinates: x (right), y (front), z (up)
        ori_points = points.copy()
        ori_feats = np.ones([ori_points.shape[0], 1], np.float32)

        coords, points = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_x=self.flip_x, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['aug_points'] = points[idxs]
        out_dict['feats'] = np.ones([len(idxs), 1], np.float32)  # simply use 1 as feature
        out_dict['seg_label'] = seg_label[idxs]
        out_dict['lidar_path'] = lidar_path

        out_dict['img_indices'] = out_dict['img_indices'][idxs]

        if self.ps_label_dir is not None:
            ps_label_2d = refine_pseudo_labels(
                data_dict['probs_2d'],
                data_dict['pseudo_label_2d'].astype(np.int32)
            )
            ps_label_3d = refine_pseudo_labels(
                data_dict['probs_3d'],
                data_dict['pseudo_label_3d'].astype(np.int32)
            )
            out_dict.update({
                'pseudo_label_2d': ps_label_2d[idxs],
                'pseudo_label_3d': ps_label_3d[idxs],
                'ori_pseudo_label_3d': ps_label_3d
            })

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'orig_points_idx': idxs,
            })
        
        if 'g_indices' in data_dict.keys():
            g_mask = data_dict['g_indices']
            out_dict.update({'g_indices': g_mask})

        if self.ema_input:
            ori_coords, _ = augment_and_scale_3d(ori_points, self.scale, self.full_scale)
            ori_idxs = (ori_coords.min(1) >= 0) * (ori_coords.max(1) < self.full_scale)
            out_dict['ori_img_indices'] = ori_points_img.astype(np.int64)
            out_dict['ori_img'] = np.moveaxis(ori_image, -1, 0)
            out_dict['ori_coords'] = ori_coords[ori_idxs]
            out_dict['ori_feats'] = np.ones([out_dict['ori_coords'].shape[0], 1], np.float32)
            out_dict['aug_keep_idx'] = np.ones(ori_points.shape[0], dtype=np.bool8)
            out_dict['ori_idxs'] = idxs
        
        if self.use_pc_mm:
            out_dict.update({
                'ori_obj_pc': data_dict['ori_obj_pc'],
                'ori_obj_label': data_dict['ori_obj_label'],
                'ori_img_size': data_dict['ori_img_size'],
                'proj_matrix': data_dict['proj_matrix'],
                'ori_points': ori_points,
                'ori_feats': ori_feats
            })
        
        if 'sam_mask' in data_dict.keys():
            out_dict['sam_mask'] = data_dict['sam_mask']
        
        return out_dict


def test_NuScenesSCN():
    from mopa.data.utils.visualize import draw_points_image_labels, \
        draw_bird_eye_view, show_anns
    from tqdm import tqdm
    preprocess_dir = "mopa/datasets/nuscenes/preprocess/preprocess"
    nuscenes_dir = "mopa/datasets/nuscenes"
    # # split = ('train_singapore',)
    # # pselab_paths = ('/home/docker_user/workspace/outputs/mopa/nuscenes/usa_singapore/mopa/pselab_data/train_singapore.npy',)
    # # split = ('train_night',)
    # # pselab_paths = ('/home/docker_user/workspace/outputs/mopa/nuscenes/day_night/mopa/pselab_data/train_night.npy',)
    # split = ('val_night',)
    split = ('train_singapore',)
    dataset = NuScenesSCN(
                        split=split,
                        preprocess_dir=preprocess_dir,
                        nuscenes_dir=nuscenes_dir,
                        label_mode="segmentation",
                        ps_label_dir="ps_label/0801_pslabels",
                        merge_classes=False,
                        noisy_rot=0.1,
                        flip_x=0.5,
                        transl=True,
                        fliplr=0.5,
                        color_jitter=(0.4, 0.4, 0.4),
                        sam_mask_dir='img_mask',
                        g_indices_dir='g_indices',
                        obj_root_dir='mopa/datasets/waymo/waymo_extracted/objects',
                        use_pc_mm=True,
                        obj_name_ls=["truck", "person", "bicycle", "motorcycle"]
                        )
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        coords = data['coords']
        seg_label = data['seg_label']
        
        # 1. Check point-image mapping
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        # draw_points_image_labels(img, img_indices, seg_label, color_palette_type='NuScenesLidarSeg', point_size=3, save="mopa/samples/nuscenes_sample.jpg")
        # pseudo_label_2d = data['pseudo_label_2d']
        # draw_points_image_labels(img, img_indices, pseudo_label_2d, color_palette_type='NuScenes', point_size=3)
        # draw_bird_eye_view(coords)5
        # print('Number of points:', len(coords))
        # input("Press Enter to continue...")
        
        # 2. Check g_indices validity
        g_indices = data['g_indices']
        from mopa.data.utils.visualize import debug_visualizer, \
            A2D2_COLOR_PALETTE_SHORT, NUSCENES_LIDARSEG_COLOR_PALETTE, draw_points_image_labels
        pc_color = np.ones((data['ori_points'].shape[0], 3)) * 200
        pc_color[g_indices, 2] = 0
        save_pth = "mopa/samples/nusc_ground.pcd"
        # a = debug_visualizer(data['ori_points'], save_pth, pc_color)
        # input("Pending continue....")
        
        # 4. Check SAM mask validity
        img = data['img'] * 255
        img = np.transpose(img.astype(np.uint8), axes=[1,2,0])
        sam_mask = data['sam_mask']
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        show_anns(sam_mask, img)
        plt.axis('off')
        plt.savefig("mopa/samples/sam_sample.jpg")
        
        # 3. Check pslabels validity
        assert data['pseudo_label_2d'].shape[0] == data['coords'].shape[0]
        assert data['pseudo_label_3d'].shape[0] == data['coords'].shape[0]
        draw_points_image_labels(img, img_indices, seg_label, 
                                 color_palette_type="NuScenesLidarSeg_RAW",
                                 save='mopa/samples/nuscenes_pc2img.png',
                                 point_size=20)
        
        input("Press Enter to continue...")


def compute_class_weights():
    preprocess_dir = "mopa/datasets/nuscenes/preprocess/preprocess"
    split = ('train_day', 'test_day')  # nuScenes-lidarseg USA/Singapore
    # split = ('train_day', 'test_day')  # nuScenes-lidarseg Day/Night
    # split = ('train_singapore', 'test_singapore')  # nuScenes-lidarseg Singapore/USA
    # split = ('train_night', 'test_night')  # nuScenes-lidarseg Night/Day
    # split = ('train_singapore_labeled',)  # SSDA: nuScenes-lidarseg Singapore labeled
    nuscenes_dir = "mopa/datasets/nuscenes"
    dataset = NuScenesBase(split,
                        preprocess_dir=preprocess_dir,
                        label_mode="segmentation",
                        merge_classes=False)
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


def compute_stats():
    preprocess_dir = 'path/to/data/nuscenes_lidarseg_preprocess/preprocess'
    nuscenes_dir = 'path/to/data/nuscenes'
    outdir = 'path/to/data/nuscenes_lidarseg_preprocess/stats'
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = ('train_day', 'test_day', 'train_night', 'val_night', 'test_night',
              'train_usa', 'test_usa', 'train_singapore', 'val_singapore', 'test_singapore')
    for split in splits:
        dataset = NuScenesSCN(
            split=(split,),
            preprocess_dir=preprocess_dir,
            nuscenes_dir=nuscenes_dir
        )
        # compute points per class over whole dataset
        num_classes = len(dataset.class_names)
        points_per_class = np.zeros(num_classes, int)
        for i, data in enumerate(dataset.data):
            print('{}/{}'.format(i, len(dataset)))
            points_per_class += np.bincount(data['seg_labels'], minlength=num_classes)

        plt.barh(dataset.class_names, points_per_class)
        plt.grid(axis='x')

        # add values right to the bar plot
        for i, value in enumerate(points_per_class):
            x_pos = value
            y_pos = i
            if dataset.class_names[i] == 'driveable_surface':
                x_pos -= 0.25 * points_per_class.max()
                y_pos += 0.75
            plt.text(x_pos + 0.02 * points_per_class.max(), y_pos - 0.25, f'{value:,}', color='blue', fontweight='bold')
        plt.title(split)
        plt.tight_layout()
        # plt.show()
        plt.savefig(outdir / f'{split}.png')
        plt.close()


if __name__ == '__main__':
    # test_NuScenesSCN()
    compute_class_weights()
    # compute_stats()