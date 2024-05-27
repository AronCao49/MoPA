import copy
import os.path as opt

import numpy as np
import open3d as o3d
import faiss
import torch
import torch.nn.functional as F


def draw_registration_result(sample_ls, reg_p2l_ls):
    t_sample_ls = []
    color_ls = [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
    for i in range(0, len(sample_ls)):
        source_temp = copy.deepcopy(sample_ls[i])
        # transform np.ndarray to PointCloud
        if type(source_temp) is np.ndarray:
            curr_pc = o3d.geometry.PointCloud()
            curr_pc.points = o3d.utility.Vector3dVector(source_temp[:, :3])
            # curr_pc = curr_pc.random_down_sample(0.04)
            curr_pc.estimate_normals()
            t_sample_ls.append(curr_pc)
        else:
            t_sample_ls.append(source_temp)
        if i == (len(sample_ls) - 1):
            break
        transformation = copy.deepcopy(reg_p2l_ls[i])
        t_sample_ls = [source_temp.transform(transformation) for source_temp in t_sample_ls]

    for i in range(len(t_sample_ls)):
        t_sample_ls[i].paint_uniform_color(color_ls[i])
    print(len(t_sample_ls))
    o3d.visualization.draw_geometries(t_sample_ls,
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def pose_computation(pc_array_ls):
    """
    Estimate pose between consecutive point cloud arrays
    Args:
        pc_array_ls: list of consecutive point cloud array [pc0, pc1, pc2]
    Returns:
        reg_p2l_ls: list of pose matrices [tf-0->1, tf-1->2]
    """
    pc_sample_ls = []
    reg_p2l_ls = []
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    for pc_array in pc_array_ls:
        # load specified point cloud frame
        curr_pc_array = pc_array[pc_array[:,0] > 0]

        # transform point cloud from np array to o3d point cloud
        curr_pc = o3d.geometry.PointCloud()
        curr_pc.points = o3d.utility.Vector3dVector(curr_pc_array[:, :3])
        curr_pc = curr_pc.random_down_sample(0.04)
        curr_pc.estimate_normals()
        # curr_pc.estimate_covariances()
        pc_sample_ls.append(curr_pc)

    for i in range(1,len(pc_sample_ls)):
        # provide init transition 
        trans_init = np.eye(4)
        trans_init[0,3] = 1.0
        threshold = 2
        curr_pc = pc_sample_ls[i-1]
        next_pc = pc_sample_ls[i]
        
        # begin ICP estimation
        reg_p2l = o3d.pipelines.registration.registration_icp(
                    curr_pc, next_pc, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    # o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                    )
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        reg_p2l_ls.append(reg_p2l.transformation)
    return reg_p2l_ls

def knn_search(cfg, data_batch):
    """
    Function to perform KNN search based on L2 norm distance of points
    Args:
        cfg: config dictionary
        data_batch: data dictionary that contains information of inputs
    Return:
        idx_ls: list of KNN indices for each sample
        mspc_idx_ls: list of len of multi-scan pc
    """
    # compute pose and map multi-scan pc
    start_idx =  0
    # allocated list for semantic logits and aggregated pc
    query_pc_ls = []
    mscan_pc_ls = []
    mspc_idx_ls = []
    
    for i in range(len(data_batch['seq_length'])):
        # allocate list of transformed pc
        seq_len = data_batch['seq_length'][i]
        ori_idx = data_batch['ori_idx'][i]
        tf_pc = []
        
        # Option 1: compute pose online, using ICP from open3d
        if "CARTIN" in cfg.DATASET_TARGET.TYPE.upper():
            pc_array_ls = data_batch['points'][start_idx:start_idx+seq_len]
            keep_idx_sls = data_batch['keep_idx'][start_idx:start_idx+seq_len]
            pose_tf = pose_computation(pc_array_ls)
            mscan_pc_num = keep_idx_sls[-1].sum()
            # build multi-scan pc
            for i in range(0,seq_len-1):
                tf = pose_tf[i]
                front_pc = pc_array_ls[i][keep_idx_sls[i]]
                front_pc = np.concatenate([front_pc, np.ones([front_pc.shape[0], 1], dtype=np.float32)], axis=1)
                tf_pc.append(front_pc)
                tf_pc = [(tf @ pc.T).T for pc in tf_pc]
                mscan_pc_num += keep_idx_sls[i].sum()
            query_pc_ls.append(pc_array_ls[-1][keep_idx_sls[-1]])
            tf_pc.append(pc_array_ls[-1][keep_idx_sls[-1]])
            tf_pc = [pc[:, :3] for pc in tf_pc]
        
        # Option 2: using gt pose
        else:
            # retrieve pc array & keep idx of current seq
            pc_array_ls = data_batch['points'][start_idx:start_idx+seq_len]
            # preserve xyz from transformed pc
            query_pc_ls.append(pc_array_ls[ori_idx])
            mscan_pc_num = 0
            for pc in pc_array_ls:
                tf_pc.append(pc)
                mscan_pc_num += pc.shape[0]
            
        # check whether length of tf_pc equal to seq_len
        assert len(tf_pc) == seq_len
        mscan_pc_ls.append(np.concatenate(tf_pc, axis=0))
        # update start idx
        start_idx += seq_len
        mspc_idx_ls.append(mscan_pc_num)

    # TODO: KNN search nearest neighbors & voting mechanism
    idx_ls = []
    for i in range(len(query_pc_ls)):
        # retrieve query and its corresponding mscan
        query_pc = query_pc_ls[i].astype(np.float32)
        mscan_pc = mscan_pc_ls[i].astype(np.float32)
        # create indexes and add mscan
        index = faiss.IndexFlatL2(3)
        index.add(np.ascontiguousarray(mscan_pc))
        # KNN search based on L2 distance
        k = cfg.TRAIN.XMUDA.knn_k
        _, idx = index.search(np.ascontiguousarray(query_pc), k)
        idx = torch.from_numpy(idx)
        idx_ls.append(idx)

    return idx_ls, mspc_idx_ls
    

if __name__ == "__main__":
    demo_seq_pth = "mopa/datasets/NTU_v2/bin/viral__2022-02-13-13-48-34"
    sample_num = 1000
    rl_frame_ls = [-2, -1, 0]
    pc_sample_ls = []
    reg_p2l_ls = []

    for rl_frame in rl_frame_ls:
        # load specified point cloud frame
        sample = "frame{:06d}.bin".format(sample_num + rl_frame)
        curr_pc_array = np.fromfile(opt.join(demo_seq_pth, sample), dtype=np.float32)
        curr_pc_array = curr_pc_array.reshape((-1,4))
        # curr_pc_array = curr_pc_array[curr_pc_array[:,0] > 0]

        # transform point cloud from np array to o3d point cloud
        curr_pc = o3d.geometry.PointCloud()
        curr_pc.points = o3d.utility.Vector3dVector(curr_pc_array[:, :3])
        curr_pc = curr_pc.voxel_down_sample(voxel_size=0.3)
        curr_pc.estimate_normals()
        # curr_pc.estimate_covariances()
        pc_sample_ls.append(curr_pc)

    for i in range(1,len(pc_sample_ls)):
        # provide init transition 
        trans_init = np.eye(4)
        trans_init[0,3] = 1.0
        threshold = 3
        curr_pc = pc_sample_ls[i-1]
        next_pc = pc_sample_ls[i]
        
        # begin ICP estimation
        print("Begin point-to-plane ICP")
        reg_p2l = o3d.pipelines.registration.registration_icp(
                    curr_pc, next_pc, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    # o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
                    )
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        reg_p2l_ls.append(reg_p2l.transformation)

    draw_registration_result(pc_sample_ls, reg_p2l_ls)
