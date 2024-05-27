import argparse
from copy import deepcopy
import open3d as o3d
import os
import numpy as np
import scipy
import random

from sklearn.cluster import DBSCAN
from tqdm import tqdm
from progress.bar import Bar

from mopa.data.utils.visualize import debug_visualizer

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

color_map = np.asarray([
    (255, 179, 0),
    (128, 62, 117),
    (255, 104, 0),
    (166, 189, 215),
    (193, 0, 32),
    (206, 162, 98),
    (129, 112, 102),

    # these aren't good for people with defective color vision:
    (0, 125, 52),
    (246, 118, 142),
    (0, 83, 138),
    (255, 122, 92),
    (83, 55, 122),
    (255, 142, 0),
    (179, 40, 81),
    (244, 200, 0),
    (127, 24, 13),
    (147, 170, 0),
    (89, 51, 21),
    (241, 58, 19),
    (35, 44, 22)])


def object_point_extraction(
        src_data_dir: str,
        obj_class_id: list,
        obj_class_name: list, 
        save_dir: str,
        max_num: int,
        max_distance: 10) -> int:
    """Object points extraction

    Extract instances of object point clouds given the desired class ID list.
    Instances are separated based on DBSCAN. Currently only design for Waymo

    Args:
        src_root_dir: 
            path to point cloud data of the source dataset for point cloud 
            extraction. Should only include the TRAINING data only.
        obj_class_id: 
            list of INT class ID for extraction.
        obj_class_name: 
            list of corresponding class names of obj_class_id
        save_dir: 
            path to the saveing dir of extraction results.
        max_num: 
            maximum number of stored instances.
    
    Returns:
        NULL
    """
    inst_num_count = [0 for i in range(len(obj_class_id))]
    bar = Bar("Object Extraction", max=(len(obj_class_id) * max_num))

    for sequence in os.listdir(src_data_dir):
        seq_dir = os.path.join(src_data_dir, sequence)
        pc_dir = os.path.join(seq_dir, "bin")         # lidar dir
        label_dir = os.path.join(seq_dir, "label")      # label dir
        
        for pc_file in os.listdir(pc_dir):
            # skip not-bin file
            if ".bin" not in pc_file:
                continue

            pc = np.fromfile(os.path.join(pc_dir, pc_file), dtype=np.float32).reshape(-1, 4)
            label_file = pc_file.replace('.bin', '.npy')
            label = np.load(os.path.join(label_dir, label_file)).astype(np.int64)

            for i in range(len(obj_class_id)):
                # specify class id and name, create dir
                class_id = obj_class_id[i]
                class_name = obj_class_name[i]
                object_dir = os.path.join(save_dir, class_name)
                if not os.path.exists(object_dir):
                    os.mkdir(object_dir)

                # filter out samples
                class_idx = label == class_id
                class_pc = pc[class_idx, :]
                if class_pc.shape[0] == 0:
                    continue
                range_cls_pc = np.sqrt(class_pc[:, 0] * class_pc[:, 0] + \
                                       class_pc[:, 1] * class_pc[:, 1] + \
                                       class_pc[:, 2] * class_pc[:, 2] + 0.00001)

                # DBSCAN clustering
                inst_cluster = DBSCAN(eps=4, min_samples=5)
                test_pc = deepcopy(class_pc[:, :3])
                inst_label = inst_cluster.fit_predict(test_pc)

                # filter out cluster center locate too far
                inrange_pc = []
                for inst_id in np.unique(inst_label).tolist():
                    if inst_num_count[i] >= max_num:    # limit obj num
                        continue
                    inst_center = np.average(range_cls_pc[inst_label == inst_id])
                    if inst_center <= max_distance:
                        inrange_pc = class_pc[inst_label == inst_id, :]
                        inst_num_count[i] = inst_num_count[i] + 1
                        # save objects
                        save_file_pth = os.path.join(
                            object_dir, 
                            "{:05d}.bin".format(inst_num_count[i])
                            )
                        save_pc = inrange_pc.astype(np.float32)
                        save_pc.tofile(save_file_pth)
                        bar.next()
                
                if sum(inst_num_count) >= len(obj_class_id) * max_num:
                    bar.finish()
                    return 0
    
    print("Does not contain enough objects")
    return 0


def obj_bin_to_pcd(
    bin_dir: str,
    pcd_dir: str
) -> int:
    """Function to save bin files to pcd files

    Args:
        bin_dir (str): path to dir that stores bin files
        pcd_dir (str): path to dir that saves pcd files

    Returns:
        int: simple return
    """
    for bin_file in tqdm(sorted(os.listdir(bin_dir))):
        bin_pth = os.path.join(bin_dir, bin_file)
        pcd_path = os.path.join(pcd_dir, bin_file.replace(".bin", ".pcd"))
        pc = np.fromfile(bin_pth, dtype=np.float32).reshape(-1,4)[:, :3]
        a = debug_visualizer(pc, np.ones((pc.shape[0], 3)), pcd_path)
    
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Waymo Data Extraction')
    parser.add_argument('--task', default="MoPA", type=str, help='task name')
    parser.add_argument('--waymo_dir',
                        default="mopa/datasets/waymo/waymo_extracted/training",
                        type=str, help='path to directory with waymo extracted data')
    parser.add_argument('--obj_save_dir',
                        default="mopa/datasets/waymo/waymo_extracted/objects",
                        type=str, help='path to directory with extracted ROs')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_random_seeds()
    
    args = parse_args()
    
    # Extract object points
    src_data_dir = args.waymo_dir
    
    obj_class_id = [12, 13, 7]
    obj_class_name = ["bicycle", "motorcycle", "pedestrian"]
    
    save_dir = args.obj_save_dir
    max_num = 1000
    max_distance = 15

    idx = object_point_extraction(
        src_data_dir, 
        obj_class_id, 
        obj_class_name, 
        save_dir, 
        max_num,
        max_distance
        )