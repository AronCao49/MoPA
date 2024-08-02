from typing import List
import torch
import sys
import time

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
import torch.nn.functional as F

from mopa.common.utils.loss import l2_norm
from mopa.data.utils.refine_pseudo_labels import refine_negative_voxels

# patchwork
patchwork_module_path = "mopa/third_party/patchwork-plusplus/build/python_wrapper"
sys.path.insert(0, patchwork_module_path)
import pypatchworkpp

from mopa.data.utils.augmentation_3d import augment_and_scale_3d, range_projection
from mopa.data.utils.visualize import debug_visualizer, draw_range_image_labels


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

def point_mixmatch(
        ori_pc: np.ndarray,
        ori_label: np.ndarray,
        obj_pc_ls: List[np.ndarray],
        obj_label_ls: List[np.ndarray],  
        z_disc: np.float32 = -0.324, 
        obj_aug: dict = None,
        insert_mode: str = "ground",
        # grounding insertation
        search_voxel_size: float = 0.5,
        search_range: list = [50, 50],
        search_z_min: float = -2.0,
        proj_matrix: np.ndarray = None,
        image_size: tuple = (),
        g_indices: np.ndarray = None,
        front_axis: str = 'x'
        ): 
    """
    Function of object-based point mix-match
    Args:
        ori_pc:
            the unaugmented point cloud as array of [N, 4].
        obj_pc_ls:
            the adding object point cloud list of arrays of [N, 4].
        z_discr:
            vertical discripency between sensors of two pc 
            (ori_pc_sensor - obj_pc_sensor).
        obj_aug:
            dictionaries to use augmentations for obj_pc, including:
                e.g., noisy (TODO);
        fv_mode:
            whether to force the object located at front (x>0)
    Return:
        [cat_pc, cat_label]:
            1. cat_pc:
                concatenated point clouds (ori_pc + obj_pc)
            1. cat_label:
                concatenated labels (ori_label + obj_label)
    """
    new_obj_pc = []
    new_obj_label = []
    if insert_mode == "fv":
        # align the lowest point between ori_pc and obj_pc
        for i in range(len(obj_pc_ls)):
            obj_pc = obj_pc_ls[i]
            obj_pc[:, 2] = obj_pc[:, 2] - z_disc

            # front_view rotation
            obj_pc_ctr = np.average(obj_pc, axis=0)
            if obj_pc_ctr[0] < 0:
                xy_theta = np.arccos(
                    obj_pc_ctr[1] / np.sqrt(obj_pc_ctr[0]**2 + obj_pc_ctr[1]**2)
                    )        # theta on xy plane between point and x axis
                fv_rot_matrix = np.array([
                    [np.cos(2 * xy_theta), -np.sin(2 * xy_theta), 0],
                    [np.sin(2 * xy_theta), np.cos(2 * xy_theta), 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                obj_pc[:, :3] = obj_pc[:, :3].dot(fv_rot_matrix)
            
            # pass as valid obj
            new_obj_label.append(obj_label_ls[i])
            new_obj_pc.append(obj_pc)
        pass_index = 0
    
    elif insert_mode == "ground":
        # Consider multiple objects in obj_pc_ls
        # Step 1: find the obj with largest x/y extent as overlap checking anchor
        all_obj_extents = np.zeros(len(obj_pc_ls))
        for i in range(len(obj_pc_ls)):
            obj_pc = obj_pc_ls[i]
            # extent_xy = np.max(
               #     np.max(obj_pc, axis=0)[0:2] - np.min(obj_pc, axis=0)[0:2]
              # )
            extent_xy = np.linalg.norm(np.max(obj_pc, axis=0)[0:2] - np.min(obj_pc, axis=0)[0:2])
            all_obj_extents[i] = extent_xy
        anchor_obj_idx = np.argsort(all_obj_extents)[::-1]
  
        # Step 2: use different anchor obj to compute valid insertation centers
        # until there is valid centers at the end
        valid_obj_ls = obj_pc_ls.copy()
        ignore_idx_ls = []
        pass_index = 1
        for idx_i in range(len(anchor_obj_idx)):
            obj_idx = anchor_obj_idx[idx_i]
            obj_pc = obj_pc_ls[obj_idx]

            # Condition 0: non-overlap
            valid_centers = check_overlap(
                ori_pc, obj_pc[:, :3],
                voxel_size = search_voxel_size,
                search_range = search_range,
                z_min = search_z_min,
                z_max = None,
                front_axis = front_axis
            )
            # continue if no valid insertion for current object
            if valid_centers is None:
                ignore_idx_ls.append(idx_i)
                continue

            # Condition 1: preserve valid centers in the fov
            keep_idx = valid_centers[:, 0] > 0
            valid_centers = valid_centers[keep_idx]
            valid_img_centers = np.concatenate(
                (valid_centers, np.ones((valid_centers.shape[0], 1))), axis=1)
            valid_img_centers = np.matmul(
                proj_matrix.astype(np.float32),
                valid_img_centers.astype(np.float32).T, dtype=np.float32).T
            valid_img_centers = valid_img_centers[:, :2] / np.expand_dims(valid_img_centers[:, 2], axis=1)
            vc_indices = select_points_in_frustum(valid_img_centers, 0, 0, *image_size)
            valid_centers = valid_centers[vc_indices]
            
            # Condition 2: preserve only centers located further than the original obj center
            obj_center = (np.max(obj_pc, axis=0) + np.min(obj_pc, axis=0)) / 2
            ori_range = np.sqrt(np.square(obj_center[0]) + np.square(obj_center[1]))
            valid_range = np.sqrt(
                np.square(valid_centers[:, 0]) + np.square(valid_centers[:, 1]))
            valid_centers = valid_centers[valid_range >= ori_range]
            
            # continue if no valid insertion for current object
            if valid_centers.shape[0] == 0:
                ignore_idx_ls.append(idx_i)
                continue
            
            # Step 3: preserve the valid centers on the ground & random sample
            tr_mtx_ls = obj_on_road(
                ori_pc, valid_obj_ls, valid_centers, 
                voxel_size = search_voxel_size, g_mask=g_indices
            )
   
            # continue if no valid insertion for current object
            if tr_mtx_ls is None:
                ignore_idx_ls.append(idx_i)
                continue

            # Step 4: style translation through ray chasing
            pass_index = 0
            new_obj_pc = []
            new_obj_label = []
            for i in range(len(valid_obj_ls)):
                if i in ignore_idx_ls:
                    continue    # ignore invalid obj pc
                # Transform valid obj_pc
                obj_pc_xyz = np.concatenate(
                    (valid_obj_ls[i][:, :3], \
                        np.ones((valid_obj_ls[i].shape[0], 1))), axis=1)
                obj_pc_xyz = (tr_mtx_ls[i] @ obj_pc_xyz.T).T
                new_obj_pc.append(obj_pc_xyz[:, :3])
                new_obj_label.append(obj_label_ls[i])
            break        # break if exist valid solutions

    if pass_index == 0:
        obj_pc = np.concatenate(new_obj_pc, axis=0)
        obj_label = np.concatenate(new_obj_label, axis=0)
        cat_pc = np.concatenate((ori_pc[:, :3], obj_pc[:, :3]), axis=0)
        cat_label = np.concatenate((ori_label, obj_label), axis=0)
        obj_mask = np.zeros(cat_pc.shape[0])
        obj_mask[-obj_pc.shape[0]:] = 1
        obj_mask = obj_mask.astype(np.bool8)
        obj_ps_mask = np.zeros(cat_label.shape[0])
        obj_ps_mask[-obj_pc.shape[0]:] = 1
        obj_ps_mask = obj_ps_mask.astype(np.bool8)

    else:
        cat_pc = ori_pc
        cat_label = ori_label
        obj_mask = np.zeros(ori_pc.shape[0], dtype=np.bool8)
        obj_ps_mask = np.zeros_like(obj_mask, dtype=np.bool8)
    
    return cat_pc, cat_label, obj_mask, obj_ps_mask


def check_overlap(
    pc_scan: np.ndarray,
    pc_obj: np.ndarray,
    # Voxel-based arguments
    voxel_size: float = 0.2,
    search_range: list = [25.0, 25.0],
    z_min: float = -2.0,        # for kitti filter out noisy points
    z_max: float = None, 
    front_axis: str = "x",
) -> list:
    """Function based on torch conv to check overlap
    There are two ways to do that:
        1. Voxel based checking, which need more conservative overlap checking.
           The voxel based would not work, considering some objects are too large.
        2. Cylinder based checking, which need more complicated partition methods.
           This method need to recompute the cylinder extent (dr, dtheta, dz) 
           based on the object location

    Args:
        pc_scan (np.ndarray): numpy array of the current lidar scan (x, y, z)
        pc_obj (np.ndarray): the points of the object (x, y, z)
        r_step (float): step for range iteration
        z_step (float): step for z iteration
        search_range (list): searching range for object placement (x, y)
    Return:
        center_ls (list): list of availabel centers (x, y, z)
    """
    start_time = time.time()
    # voxelize for scan and objects
    _, pc_indices = sparse_quantize(pc_scan[:, :3], voxel_size=voxel_size, return_index=True)
    _, obj_indices = sparse_quantize(pc_obj[:, :3], voxel_size=voxel_size, return_index=True)

    pc_op_voxels = np.floor(pc_scan[:, :3][pc_indices] / voxel_size)
    obj_op_voxels = np.floor(pc_obj[:, :3][obj_indices] / voxel_size)

    # init search range
    search_range = [
        int(search_range[0] / voxel_size), 
        int(search_range[1] / voxel_size)
    ]
 
    z_min = np.floor(z_min / voxel_size)
    z_max = z_min if z_max is None else z_max
    extent_z = np.max(obj_op_voxels, axis=0)[2] - \
        np.min(obj_op_voxels, axis=0)[2] + 2
    search_range.append(int(extent_z + z_max)) 
 
    voxel_grid = np.zeros(
        (2 * search_range[0], 2 * search_range[1], int(search_range[2] - z_min)))
 
    # Update pc_grid (need to specify front direction axis)
    if front_axis == 'x':
        x_idxs = np.logical_and(
            pc_op_voxels[:, 0] >= 0, 
            pc_op_voxels[:, 0] < 2 * search_range[0]
        )
        y_idxs = np.logical_and(
            pc_op_voxels[:, 1] >= -search_range[1], 
            pc_op_voxels[:, 1] < search_range[1]
        )
        v2g_offset = np.array([0, -search_range[1], z_min])
    elif front_axis == 'y':
        x_idxs = np.logical_and(
            pc_op_voxels[:, 0] >= -search_range[0], 
            pc_op_voxels[:, 0] < search_range[0]
        )
        y_idxs = np.logical_and(
            pc_op_voxels[:, 1] >= 0, 
            pc_op_voxels[:, 1] < 2 * search_range[1]
        )
        v2g_offset = np.array([-search_range[0], 0, z_min])
    z_idxs = np.logical_and(
        pc_op_voxels[:, 2] >= z_min,
        pc_op_voxels[:, 2] < search_range[2]
    )

    # Index occupied voxel in voxel grid
    xyz_idxs = np.logical_and(x_idxs, y_idxs)
    xyz_idxs = np.logical_and(xyz_idxs, z_idxs)
    if np.any(xyz_idxs):            # Avoid non-occupancy
        voxel_in_range = pc_op_voxels[xyz_idxs, :]
        voxel_in_range = (voxel_in_range - v2g_offset).astype(np.int32)
        voxel_grid[voxel_in_range[:, 0], 
                voxel_in_range[:, 1], 
                voxel_in_range[:, 2]] = 1

    # Update obj_cub with circumscribed circle for potential rotation
    obj_voxel_extent = np.max(obj_op_voxels, axis=0) - \
        np.min(obj_op_voxels, axis=0) + 1
    obj_voxel_extent[0:2] = np.ceil(np.sqrt(
        np.square(obj_voxel_extent[0]) + np.square(obj_voxel_extent[1])))
    obj_cub = np.ones(obj_voxel_extent.astype(np.int32).tolist())
 
    # Convert all cpu array to gpu tensors and compute overlap check
    # input: (1, 1, X, Y, Z); cub: (1, 1, dx, dy, dz)
    with torch.no_grad():
        obj_cub = torch.from_numpy(obj_cub).cuda().unsqueeze(0).unsqueeze(0)
        voxel_grid = torch.from_numpy(voxel_grid).cuda().unsqueeze(0).unsqueeze(0)
        overlap_results = F.conv3d(voxel_grid, obj_cub)
        # Trace back from output == 0 to inputs
        valid_output_idxs = torch.nonzero(overlap_results.squeeze(0).squeeze(0) == 0)
        valid_output_idxs = valid_output_idxs.cpu().numpy()

    # Retrieve the corresponding and compute the center
    if valid_output_idxs.shape[0] > 0:
        x_start_input, y_start_input, z_start_input = \
            valid_output_idxs[:, 0], valid_output_idxs[:, 1], valid_output_idxs[:, 2]
        # Output mapped back to input, not using // since center can be float in this case
        x_center = x_start_input + (obj_voxel_extent[0] - 1) / 2
        y_center = y_start_input + (obj_voxel_extent[1] - 1) / 2
        z_center = z_start_input + (obj_voxel_extent[2] - 1) / 2
        valid_centers = np.column_stack((x_center, y_center, z_center))
        valid_centers = (valid_centers + v2g_offset) * voxel_size
    else:
        valid_centers = None
 
    return valid_centers


def cartesian_to_cylinder(center: np.ndarray) -> np.array:
    """Function to convert 2D Cartesian center to Cylinder coord

    Args:
        center (np.ndarray): the center under Cartesian coord (x, y)

    Returns:
        np.ndarray: the Cylidner center (r, theta), theta in [-pi, pi]
    """
    center_cld = np.array([
        np.sqrt(np.square(center[0]) + np.square(center[1])),
        np.arctan(center[1] / center[0])
    ])
    # from -pi/2:pi/2 to -pi:pi
    if center[0] < 0 and center[1] < 0:
        center_cld[1] -= np.pi
    if center[0] < 0 and center[1] > 0:
        center_cld[1] += np.pi
    return center_cld


def obj_on_road(
    ori_pc: np.ndarray,
    obj_pc_ls: List[np.ndarray],
    valid_centers: np.ndarray,
    voxel_size: float = 0.5,
    # Offline g_indices
    g_mask: np.ndarray = None,
) -> List[np.ndarray]:
    """Function to place object on the road and refine

    Args:
        ori_pc (np.ndarray): Original scan (N, 4).
        obj_pc_ls (list of np.ndarray): List of original object points (N, 3).
        valid_centers (np.ndarray): Collection of valid centers (N_c, 3).
        voxel_size (float, optional): Voxel size to quantization. Defaults to 0.5.

    Returns:
        List(np.ndarray): The list of Tr matrix to transform original object 
              to the augmented location.
    """
    # Quantize scan and valid centers
    _, pc_indices, pc_inverse = sparse_quantize(
        ori_pc[:, :3], voxel_size=voxel_size, return_index=True, return_inverse=True)
    voxel_centers = np.floor(valid_centers / voxel_size)
    
    # Online g_indices generation if needed
    if g_mask is None:
        params = pypatchworkpp.Parameters()
        params.verbose = False
        PatchworkPP = pypatchworkpp.patchworkpp(params)
        PatchworkPP.estimateGround(ori_pc)
        g_indices = (PatchworkPP.getGroundIndices()).astype(np.int32)
        g_mask = np.zeros(ori_pc.shape[0])
        g_mask[g_indices] = 1
  
    g_mask = g_mask[pc_indices].astype(np.bool8)     # voxelize ground mask
    voxel_pc = np.floor(ori_pc[:, :3][pc_indices] / voxel_size)    # voxelize pc
    
    # Filter out valid centers on the ground & select one xy-plane center 
    with torch.no_grad():
        road_x = torch.from_numpy(voxel_pc[g_mask][:, 0]).cuda()
        road_y = torch.from_numpy(voxel_pc[g_mask][:, 1]).cuda()
        vc_tensor = torch.from_numpy(voxel_centers).cuda()
        g_center_indices = torch.where(
            (vc_tensor[:, 0] == road_x[:, None]) & \
            (vc_tensor[:, 1] == road_y[:, None])
        )[1].cpu().numpy()

    # Return if no valid centers on the ground
    if g_center_indices.shape[0] == 0:
        return None

    g_centers_xyz = voxel_centers[g_center_indices]
    g_centers = np.unique(g_centers_xyz[:, :2], axis=0)
    rd_idx_all = np.random.choice(g_centers.shape[0], len(obj_pc_ls))
    tr_mtx_ls = []
    for i in range(rd_idx_all.shape[0]):
        rd_idx = rd_idx_all[i]
        obj_pc = obj_pc_ls[i][:,:3]
        new_center = g_centers[rd_idx, :] * voxel_size
        
        # Cylinder-based translation for oriantation preserving
        # 1. oriantation-based translation. 2. z-rotation
        obj_center = (np.max(obj_pc, axis=0) + np.min(obj_pc, axis=0)) / 2
        obj_center_cld = cartesian_to_cylinder(obj_center)
        new_center_cld = cartesian_to_cylinder(new_center)
        d_r, d_theta = new_center_cld - obj_center_cld
        xyz_disc = np.array([
            d_r * np.cos(obj_center_cld[1]), d_r * np.sin(obj_center_cld[1]), 0
        ])
        tr_mtx = np.array([
            [np.cos(d_theta), -np.sin(d_theta), 0, 0],
            [np.sin(d_theta),  np.cos(d_theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
  
        # Compensate z-discrepancy
        curr_g_mask = g_mask.copy()
        road_z_indices = np.logical_and(
            voxel_pc[g_mask][:, 0] == (new_center[0] / voxel_size),
            voxel_pc[g_mask][:, 1] == (new_center[1] / voxel_size)
        )
        curr_g_mask[curr_g_mask] = road_z_indices
        inter_g_mask = np.where(curr_g_mask)[0]
        if inter_g_mask.shape[0] > 1:
            min_z_idx = np.argmin(voxel_pc[inter_g_mask, 2], axis=0)
            inter_g_mask = inter_g_mask[min_z_idx]
        road_pc_indices = np.where(pc_inverse == inter_g_mask)[0]
        road_z = np.mean(ori_pc[road_pc_indices], axis=0)[2]
        z_disc = road_z - np.min(obj_pc[:,2], axis=0)
        xyz_disc[2] = z_disc + np.random.rand() * 0.1        # smallskip

        # Generate Tr matrix by FIRST translate THEN rotate
        t_mtx = np.eye(4)
        t_mtx[:3, 3] = xyz_disc
        tr_mtx = tr_mtx @ t_mtx
  
        tr_mtx_ls.append(tr_mtx)
    
    return tr_mtx_ls


def post_process(
    cat_pc_ls: List[np.ndarray],
    cat_pslabel_ls: List[np.ndarray],
    obj_mask_ls: List[np.ndarray],
    scale: int,
    full_scale: int,
    augment_3d: dict,
    # range image proj
    proj_W: int = 1024,
    proj_H: int = 64,
    fov_up: float = 0.05235,
    fov_down: float = -0.43633,
    scan_pth_ls: list = None,
    use_proj: bool = True,
    backbone: str = "SCN"
) -> List[torch.Tensor]:
    """Function to prepare mix-match input and labels (Only for SCN for now)
    
    Args:
        cat_pc_ls (List[np.ndarray]): Concatenated point clouds [(N, 4)].
        cat_pslabel_ls (List[np.ndarray]): Concatenated object points [(N,)].
        obj_mask_ls (List[np.ndarray]): Mask to extract obj_pc from cat_pc
        scale (int): Voxels per meter.
        full_scale (int): Size of the receptive field of SparseConvNet.
        augment_3d (dict): 3D augmentation dictionary.
        proj_W (int): width of projected range image.
        proj_H (int): height of projected range image.
        fov_up (float): upward bound of field-of-view.
        fov_down (float): downward bound of field-of-view.
        scan_pth_ls (list): path for each scan for debug.
        use_proj (bool): whether to use projection removed or not.
        backbone (str): types of 3D networks for post processing.
    
    Returns:
        List[torch.Tensor] (cat_input, cat_ps_label): prepared input and output
    """
    if "SCN" in backbone:
        locs = []
        feats = []
    else:
        raise IndexError("The specified backbone is not supported: {}".format(backbone))
    pseudo_label = []
    mask_ls = []
    aug_points_ls = []
    for i in range(len(cat_pc_ls)):
        # remove overlap points based on range image projection
        # print("Current Scan: {}".format(scan_pth_ls[i]))
        try:
            assert not np.any(np.isnan(cat_pc_ls[i][:, :3]))
        except:
            np.save('mopa/samples/bug_array.npy',cat_pc_ls[i][:, :3])
            raise AssertionError("Found Nan object points: {}".format(scan_pth_ls[i]))
        
        if use_proj and np.any(obj_mask_ls[i]):
            range_dict = range_projection(
                np.concatenate(
                    (cat_pc_ls[i][:, :3], 
                    np.ones((cat_pc_ls[i].shape[0], 1))), 
                    axis=1
                ), fov_up, fov_down, proj_W, proj_H,
                crop=False,
                obj_mask=obj_mask_ls[i]
            )
            valid_idx = range_dict['pres_idx']
        else:
            valid_idx = np.ones((cat_pc_ls[i].shape[0]), dtype=bool)
     
        # augmentation and scale
        coords, aug_points = augment_and_scale_3d(
            cat_pc_ls[i][valid_idx, :3], scale, full_scale,
            # augmentations args
            noisy_rot=augment_3d['noisy_rot'],
            flip_y=augment_3d['flip_y'] if 'flip_y' in augment_3d.keys() else 0.0,
            flip_x=augment_3d['flip_x'] if 'flip_x' in augment_3d.keys() else 0.0,
            rot_z=augment_3d['rot_z'],
            transl=augment_3d['transl']
        )
        
        # cast to integer & use only voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < full_scale)
        if "SCN" in backbone:
            coords = coords.astype(np.int64)
            coords = coords[idxs]
            # numpy to torch
            batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(i)
            locs.append(torch.cat([torch.from_numpy(coords), batch_idxs], 1))
            feats.append((torch.ones(coords.shape[0], 1)).float())
        else:
            raise IndexError("The specified backbone is not supported: {}".format(backbone))
        pseudo_label.append(torch.from_numpy(cat_pslabel_ls[i][valid_idx][idxs]))
        mask_ls.append(torch.from_numpy(obj_mask_ls[i][valid_idx][idxs]))
        aug_points_ls.append(aug_points[idxs])
    
    # concatenation
    if "SCN" in backbone:
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        cat_input = {'x': [locs, feats]}
    cat_ps_label = torch.cat(pseudo_label, 0)
    obj_mask = torch.cat(mask_ls, 0)
 
    return [cat_input, cat_ps_label, obj_mask, aug_points_ls]


if __name__ == "__main__":
    import os
    import time
    import open3d as o3d
    from mopa.data.utils.visualize import save_cuboid_centers_to_obj

    # Testing code for overlap_checking
    sample_pc_pth = "mopa/datasets/semantic_kitti/dataset/sequences/00/velodyne/000000.bin"
    obj_pc_pth = "mopa/datasets/waymo/waymo_extracted/objects/truck/00030.bin"
    output_dir = "mopa/samples/overlap"

    pc_scan = np.fromfile(sample_pc_pth, dtype=np.float32).reshape(-1,4)
    obj_pc = np.fromfile(obj_pc_pth, dtype=np.float32).reshape(-1,4)
    obj_pc = obj_pc[:, :3]
    extents = (np.max(obj_pc, axis=0) - np.max(obj_pc, axis=0)) / 2
    obj_center = (np.max(obj_pc, axis=0) + np.max(obj_pc, axis=0)) / 2

    # test overlap check
    for i in range(1):
        start_time = time.time()
        valid_centers = check_overlap(pc_scan[:, :3], obj_pc, voxel_size=0.5, search_range=[50, 50])
        print("Overlap checking completed! Take total: {}".format(time.time() - start_time))

    if valid_centers is None:
        print("No valid centers found. Terminating...")
        exit()

    # test road seg + obj placement
    for i in range(1): 
        start_time = time.time()
        tr_mtx, g_indices = obj_on_road(pc_scan, obj_pc, valid_centers)
        print("Placing object on the road completed! Total time: {}".format(time.time() - start_time))
    
    # write to samples folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # pc scan
    device = o3d.core.Device("CPU:0")
    pc_o3d = o3d.t.geometry.PointCloud(device)
    pc_o3d.point['positions'] = o3d.core.Tensor(pc_scan[:, :3], o3d.core.float32, device)
    o3d.t.io.write_point_cloud(os.path.join(output_dir, "ori_scan.pcd"), pc_o3d)
 
    # obj
    obj_o3d = o3d.t.geometry.PointCloud(device)
    obj_o3d.point['positions'] = o3d.core.Tensor(obj_pc[:, :3], o3d.core.float32, device)
    o3d.t.io.write_point_cloud(os.path.join(output_dir, "ori_obj.pcd"), obj_o3d)
 
    # obj cuiboid
    save_cuboid_pth = os.path.join(output_dir, "cuboid.obj")
    save_cuboid_centers_to_obj(valid_centers, extents, save_cuboid_pth)
 
    # final result
    final_results_pth = os.path.join(output_dir, "obj_place_results.pcd")
    tr_obj = np.concatenate((obj_pc[:, :3], np.ones((obj_pc.shape[0], 1))), axis=1)
    tr_obj = ((tr_mtx @ tr_obj.T).T)[:, :3]
    new_center = (np.max(tr_obj, axis=0) + np.max(tr_obj, axis=0)) / 2
    new_center_valid = np.ones(4)
    new_center_valid[:3] = obj_center
    new_center_valid = tr_mtx @ new_center_valid

    pc_result = np.concatenate((pc_scan[:, :3], obj_pc[:, :3], tr_obj), axis=0)
    pc_color = np.ones((pc_scan.shape[0], 3))
    pc_color[g_indices] = np.array([0, 1, 0])
    obj_color = np.zeros((obj_pc.shape[0], 3))
    obj_color[:, 0] = 1
    tr_color = np.zeros((tr_obj.shape[0], 3))
    tr_color[:, 2] = 1
    pc_color = np.concatenate((pc_color, obj_color, tr_color), axis=0)
    pc_all = o3d.t.geometry.PointCloud(device)
    pc_all.point['positions'] = o3d.core.Tensor(pc_result, o3d.core.float32, device)
    pc_all.point['colors'] = o3d.core.Tensor(pc_color, o3d.core.float32, device)
    o3d.t.io.write_point_cloud(final_results_pth, pc_all)
    

