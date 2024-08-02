import os
import os.path as osp
import numpy as np
import pickle
import sys
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.eval.detection.utils import category_to_detection_name

from mopa.data.nuscenes.nuscenes_dataloader import NuScenesBase
from mopa.data.nuscenes.projection import map_pointcloud_to_image
from mopa.data.nuscenes import splits

# patchwork
patchwork_module_path = "mopa/third_party/patchwork-plusplus/build/python_wrapper"
sys.path.insert(0, patchwork_module_path)
import pypatchworkpp

class_names_to_id = dict(zip(NuScenesBase.class_names_obj, range(len(NuScenesBase.class_names_obj))))
if 'background' in class_names_to_id:
    del class_names_to_id['background']


def preprocess(nusc, split_names, root_dir, out_dir,
               keyword=None, keyword_action=None, subset_name=None,
               location=None):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ['filter', 'exclude']

    # init dict to save
    pkl_dict = {}
    for split_name in split_names:
         pkl_dict[split_name] = []

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == 'night':
            if curr_split == 'train':
                if curr_scene_name in splits.val_night:
                    curr_split = 'val'
        if subset_name == 'singapore':
            if curr_split == 'train':
                if curr_scene_name in splits.val_singapore:
                    curr_split = 'val'

        # filter for day/night
        if keyword:
            scene_description = nusc.get("scene", sample["scene_token"])["description"]
            if keyword.lower() in scene_description.lower():
                if keyword_action == 'exclude':
                    # skip sample
                    continue
            else:
                if keyword_action == 'filter':
                    # skip sample
                    continue

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        lidar_path, boxes_lidar, _ = nusc.get_sample_data(lidar_token)
        cam_path, boxes_front_cam, cam_intrinsic = nusc.get_sample_data(cam_front_token)

        print('{}/{} {} {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path))

        sd_rec_lidar = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record_lidar = nusc.get('calibrated_sensor',
                             sd_rec_lidar['calibrated_sensor_token'])
        pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])
        sd_rec_cam = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        cs_record_cam = nusc.get('calibrated_sensor',
                             sd_rec_cam['calibrated_sensor_token'])
        pose_record_cam = nusc.get('ego_pose', sd_rec_cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        # load lidar points
        pts = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3].T

        # map point cloud into front camera image
        pts_valid_flag, pts_cam_coord, pts_img, proj_mtx = map_pointcloud_to_image(pts, (900, 1600, 3), calib_infos)
        # fliplr so that indexing is row, col and not col, row
        pts_img = np.ascontiguousarray(np.fliplr(pts_img))

        # only use lidar points in the front camera image
        pts = pts[:, pts_valid_flag]

        num_pts = pts.shape[1]
        seg_labels = np.full(num_pts, fill_value=len(class_names_to_id), dtype=np.uint8)
        # only use boxes that are visible in camera
        valid_box_tokens = [box.token for box in boxes_front_cam]
        boxes = [box for box in boxes_lidar if box.token in valid_box_tokens]
        for box in boxes:
            # get points that lie inside of the box
            fg_mask = points_in_box(box, pts)
            det_class = category_to_detection_name(box.name)
            if det_class is not None:
                seg_labels[fg_mask] = class_names_to_id[det_class]

        # convert to relative path
        lidar_path = lidar_path.replace(root_dir + '/', '')
        cam_path = cam_path.replace(root_dir + '/', '')

        # transpose to yield shape (num_points, 3)
        pts = pts.T

        # append data to train, val or test list in pkl_dict
        data_dict = {
            'points': pts,
            'seg_labels': seg_labels,
            'points_img': pts_img,  # row, col format, shape: (num_points, 2)
            'lidar_path': lidar_path,
            'camera_path': cam_path,
            'boxes': boxes_lidar,
            "sample_token": sample["token"],
            "scene_name": curr_scene_name,
            "calib": calib_infos,
            "valid_mask": pts_valid_flag,
            "proj_matrix": proj_mtx
        }
        pkl_dict[curr_split].append(data_dict)

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    for split_name in split_names:
        save_path = osp.join(save_dir, '{}{}.pkl'.format(split_name, '_' + subset_name if subset_name else ''))
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_dict[split_name], f)
            print('Wrote preprocessed data to ' + save_path)
            

def get_nuscenes_ground(root_dir: str, pickle_file: str, save_dir: str):
    """Function to use PatchWork extract & save ground indices

    Args:
        root_dir (str): root directory to load SemanticKITTI
        pickle_file (str): path to loading pickle file
        save_dir (str): sub directory to save ground indices
    """
    # Make save dir
    save_dir_path = osp.join(root_dir, save_dir)
    try:
        assert osp.exists(save_dir_path)
    except AssertionError:
        os.mkdir(osp.join(save_dir_path))
    
    # Init PatchWork++
    params = pypatchworkpp.Parameters()
    params.verbose = False
    PatchworkPP = pypatchworkpp.patchworkpp(params)
    
    # Load pickle file
    data_list = []
    with open(pickle_file, 'rb') as f:
        print("Loading pickle file: {}".format(pickle_file))
        data_list.extend(pickle.load(f))
    
    # Loop over all sequence
    for data in tqdm(data_list):
        points = np.fromfile(osp.join(root_dir, data['lidar_path']), dtype=np.float32).reshape(-1,5)[:, :3]
        seq_dir, lidar_file = data['lidar_path'].split("/")[-2], data['lidar_path'].split("/")[-1]
        # Make dir if not exists
        if not os.path.exists(osp.join(save_dir, seq_dir)):
            os.makedirs(osp.join(root_dir, save_dir, seq_dir), exist_ok=True)
        save_file_path = osp.join(root_dir, save_dir, seq_dir, lidar_file)
        
        # Ground segmentation
        PatchworkPP.estimateGround(points[:, :3])   # using intensity would cause bug
        g_indices = PatchworkPP.getGroundIndices()
        g_indices = g_indices.astype(np.int32)
        g_indices.tofile(save_file_path)
        
    print("Completed ground extraction of {}".format(pickle_file))


if __name__ == '__main__':
    root_dir = 'mopa/datasets/nuscenes'
    out_dir = 'mopa/datasets/nuscenes'
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    # nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    # Day/Night: We detect if "night" occurs in the scene description string.
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, location='boston', subset_name='usa')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, location='singapore', subset_name='singapore')
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, keyword='night', keyword_action='exclude', subset_name='day')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, keyword='night', keyword_action='filter', subset_name='night')
    
    # g_indices extraction
    get_nuscenes_ground(root_dir, 'mopa/datasets/nuscenes/preprocess/train_night.pkl', 'g_indices')
    get_nuscenes_ground(root_dir, 'mopa/datasets/nuscenes/preprocess/train_singapore.pkl', 'g_indices')
    