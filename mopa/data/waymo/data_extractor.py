import os

import tensorflow.compat.v1 as tf
import open3d as o3d
import numpy as np
from tqdm import tqdm
import cv2
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2

def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
            
        point_labels.append(sl_points_tensor.numpy())
    return point_labels

def extract_pc_img(seq_data, scen_dir):
    frame_num = 0

    # extract frame by frame        
    for data in seq_data:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if not frame.lasers[0].ri_return1.segmentation_label_compressed:
            continue
        # point cloud processing
        (range_images, camera_projections, segmentation_labels,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                        frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
        point_labels = convert_range_image_to_point_cloud_labels(
                        frame, range_images, segmentation_labels)
        points_all = points[0]                  # preserve points from top LiDAR only
        points_all = np.concatenate((points_all[:,3:6],points_all[:,1].reshape(points_all.shape[0],1)), axis=1)
        point_labels_all = point_labels[0]      # preserve points from top LiDAR only
        point_labels_all = point_labels_all[:, 1]

        lidar_dir = os.path.join(scen_dir, "lidar")
        label_dir = os.path.join(scen_dir, 'label')
        bin_dir = os.path.join(scen_dir, 'bin')
        os.makedirs(lidar_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(bin_dir, exist_ok=True)

        # save pcd
        proj_pcd = o3d.t.geometry.PointCloud()
        proj_pcd.point["positions"] = o3d.core.Tensor(points_all[:,0:3])
        intensities = np.tanh(points_all[:,3].reshape(-1,1))
        proj_pcd.point["intensities"] = o3d.core.Tensor(intensities)
        o3d.t.io.write_point_cloud(os.path.join(lidar_dir,"frame{0:06d}.pcd".format(frame_num)), proj_pcd)
        # save labels
        np.save(os.path.join(label_dir,"frame{0:06d}.npy".format(frame_num)), point_labels)
        # save bin
        points_all[:, 3] = np.tanh(points_all[:, 3])
        points_all = points_all.astype(np.float32)
        points_all.tofile(os.path.join(bin_dir,"frame{0:06d}.bin".format(frame_num)))
            
        # update frame_num
        frame_num += 1


def data_extractor(split, output_path):
    seq_list = sorted(os.listdir(split))
    for seq_file in tqdm(seq_list):
        if '.tfrecord' not in seq_file:
            continue
        # extract raw data from seq_file
        scen_dir = os.path.join(output_path, seq_file.replace(".tfrecord",""))
        if not os.path.exists(scen_dir):
            os.makedirs(scen_dir)
        seq_data = tf.data.TFRecordDataset(os.path.join(split, seq_file), compression_type='')
        extract_pc_img(seq_data, scen_dir)

    print("Complete data extraction!")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Waymo Data Extraction')
    parser.add_argument('--task', default="MoPA", type=str, help='task name')
    parser.add_argument('--raw_waymo_dir',
                        default="mopa/datasets/waymo",
                        type=str, help='path to directory with raw waymo training/val')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    waymo_root_dir = args.raw_waymo_dir
    split = os.path.join(waymo_root_dir, "training")
    output_path = os.path.join(waymo_root_dir, "waymo_extracted_test/training")
    data_extractor(split=split, output_path=output_path)

    split = split = os.path.join(waymo_root_dir, "validation")
    output_path = os.path.join(waymo_root_dir, "waymo_extracted_test/validation")
    data_extractor(split=split, output_path=output_path)
