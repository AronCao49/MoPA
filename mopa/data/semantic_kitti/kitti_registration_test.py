import numpy as np
import open3d as o3d
import os.path as osp

frame_s = "000001.bin"
frame_e = "000002.bin"
seq_dir = "/data1/semantic_kitti/dataset/sequences/07"
lidar_dir = "velodyne"
pose_pth = "poses.txt"
tr_mtx = [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, 
          -1.198459927713e-02, -7.210626507497e-03, 8.081198471645e-03, 
          -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01, 
          4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01]


# read point cloud
pc_s = np.fromfile(osp.join(seq_dir, lidar_dir, frame_s), dtype=np.float32)
pc_s = pc_s.reshape((-1, 4))[:, :3]
pc_e = np.fromfile(osp.join(seq_dir, lidar_dir, frame_e), dtype=np.float32)
pc_e = pc_e.reshape((-1, 4))[:, :3]

# read pose
with open(osp.join(seq_dir, pose_pth), "r") as file:
    pose_lines = file.readlines()
    pose_s = pose_lines[1].strip('\n').split(' ')
    pose_e = pose_lines[2].strip('\n').split(' ')

# transform pc_s to pc_3 using pose matrix
# form pose matrix
trt = np.identity(4)
trt[:3, :4] = np.asarray(tr_mtx).reshape((3,4))
pose_t_s = np.identity(4)
pose_t_s[:3, :4] = np.asarray(pose_s).reshape((3,4))
pose_t_s = np.linalg.inv(trt) @ pose_t_s @ trt

pose_t_e = np.identity(4)
pose_t_e[:3, :4] = np.asarray(pose_e).reshape((3,4))
pose_t_e = np.linalg.inv(trt) @ pose_t_e @ trt

# transform pc_s to pc_e
pc_e = np.concatenate([pc_e, np.ones([pc_e.shape[0], 1], dtype=np.float32)], axis=1)
pc_e = (np.linalg.inv(pose_t_s) @ pose_t_e @ pc_e.T).T
pc_e = pc_e[:, :3]
pc = [pc_s, pc_e]

# draw pc_s and pc_e with open3d
color_ls = [[0, 0.651, 0.929], [1.0, 0, 0]]
o3d_pc_ls = []
for i in range(len(pc)):
    source_temp = pc[i]
    curr_pc = o3d.geometry.PointCloud()
    curr_pc.points = o3d.utility.Vector3dVector(source_temp[:, :3])
    curr_pc.paint_uniform_color(color_ls[i])
    o3d_pc_ls.append(curr_pc)
# draw
o3d.visualization.draw_geometries(o3d_pc_ls,
                                  zoom=0.4459,
                                  front=[0.9288, -0.2951, -0.2242],
                                  lookat=[1.6784, 2.0612, 1.4451],
                                  up=[-0.3402, -0.9189, -0.1996])

