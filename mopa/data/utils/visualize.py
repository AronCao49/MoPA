from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mopa.data.utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data
import open3d as o3d
import copy
from PIL import Image


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

# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

NUSCENES_LIDARSEG_COLOR_PALETTE_DICT = OrderedDict([
    ('barrier', (112, 128, 144)),  # Slategrey
    ('bicycle', (220, 20, 60)),  # Crimson
    ('bus', (255, 127, 80)),  # Coral
    ('car', (255, 158, 0)),  # Orange
    ('construction_vehicle', (233, 150, 70)),  # Darksalmon
    ('motorcycle', (255, 61, 99)),  # Red
    ('pedestrian', (0, 0, 230)),  # Blue
    ('traffic_cone', (47, 79, 79)),  # Darkslategrey
    ('trailer', (255, 140, 0)),  # Darkorange
    ('truck', (255, 99, 71)),  # Tomato
    ('driveable_surface', (0, 207, 191)),  # nuTonomy green
    ('other_flat', (175, 0, 75)),
    ('sidewalk', (75, 0, 75)),
    ('terrain', (112, 180, 60)),
    ('manmade', (222, 184, 135)),  # Burlywood
    ('vegetation', (0, 175, 0)),  # Green
    ('ignore', (0, 0, 0)),  # Black
])

NUSCENES_LIDARSEG_COLOR_PALETTE = list(NUSCENES_LIDARSEG_COLOR_PALETTE_DICT.values())

NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT = [
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['car'],  # vehicle
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['driveable_surface'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['sidewalk'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['terrain'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['manmade'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['vegetation'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['ignore']
]

# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [150, 240, 255], # pole
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]

# SPVNAS color map
SEMANTIC_KITTI_COLOR_PALETTE_LONG_BGR = np.array([
    [245, 150, 100],
    [245, 230, 100],
    [150, 60, 30],
    [180, 30, 80],
    [255, 0, 0],
    [30, 30, 255],
    [200, 40, 255],
    [90, 30, 150],
    [255, 0, 255],
    [255, 150, 255],
    [75, 0, 75],
    [75, 0, 175],
    [0, 200, 255],
    [50, 120, 255],
    [0, 175, 0],
    [0, 60, 135],
    [80, 240, 150],
    [150, 240, 255],
    [0, 0, 255],
])
SEMANTIC_KITTI_COLOR_PALETTE_LONG = SEMANTIC_KITTI_COLOR_PALETTE_LONG_BGR[:, [2, 1, 0]]  # convert bgra to rgba

# classes after merging (as used in xMUDA)
WAYMO_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [255, 100, 100], # bus
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [150, 240, 255], # pole
    [0, 60, 135],   # tree trunk
    [0, 0, 255],    # traffi-sign
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
WAYMO_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in WAYMO_COLOR_PALETTE_SHORT_BGR]

WAYMO_COLOR_PALETTE_BGR = [
    [0, 0, 0], # undefined
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [255, 100, 100], # bus
    [150, 150, 100], # other_vehicle
    [150, 60, 30],  # motorcyclist
    [150, 60, 30],  # bicyclist
    [30, 30, 255],  # pedestrian
    [0, 0, 255],    # traffi-sign
    [0, 0, 255],    # traffi-light
    [150, 240, 255], # pole
    [255, 255, 50],  # construction_cone
    [150, 60, 30],  # bicycle
    [150, 60, 30],  # motorcycle
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [0, 60, 135],   # tree trunk
    [220, 220, 220], # curb
    [255, 0, 255],  # road
    [220, 220, 220], # lane_marker
    [255, 255, 50],  # other_ground
    [75, 0, 75],  # walkable
    [75, 0, 75],  # sidewalk
]
WAYMO_COLOR_PALETTE = [(c[2], c[1], c[0]) for c in WAYMO_COLOR_PALETTE_BGR]


def draw_points_image_labels(img, img_indices, seg_labels, show=True, color_palette_type='NuScenes', point_size=0.5, save=None):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSeg':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSeg_RAW':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE_SHORT
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        plt.show()

    if save is not None:
        plt.savefig(save)
        plt.close()

def depth_color(val, min_d=0, max_d=120):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val) # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 

def draw_points_image_depth(img, img_indices, depth, show=True, point_size=0.5, save=None):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    # depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    # colors = []
    # for depth_val in depth.tolist():
    #     colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    colors = depth_color(depth).tolist()
    fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100))
    # ax5.imshow(np.full_like(img, 255))
    ax.imshow(img)
    ax.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.7, s=point_size)

    ax.axis('off')

    if show:
        plt.show()
        
    if save is not None:
        plt.savefig(save)
        plt.close()

def draw_range_image_labels(proj_labels, show=True, save=False, color_palette_type='NuScenes'):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE_SHORT
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    proj_labels[proj_labels < 0] = len(color_palette) - 1
    proj_seg_colors = color_palette[proj_labels]
    plt.imshow(proj_seg_colors)
    
    if show:
        plt.show()
    if save:
        import cv2
        save_img = (proj_seg_colors * 255.).astype(np.uint8)
        save_img = np.stack((save_img[:,:,2], 
                             save_img[:,:,1],
                             save_img[:,:,0]), axis=2)
        print(save_img.shape)
        cv2.imwrite('mopa/samples/fig.jpg', save_img)


def print_projection_plt(points, color, img_size, depth=True):
    """ project converted velodyne points into camera image """

    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blank_im = np.ones(img_size, np.uint8)
    if depth:
        hsv_blkim = cv2.cvtColor(blank_im, cv2.COLOR_BGR2HSV)
        for i in range(points.shape[0]):
            # cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)
            cv2.circle(hsv_blkim, (np.int32(points[i,1]), np.int32(points[i,0])), 2, (int(color[i]),255,255),-1)
        rgb_blkim = cv2.cvtColor(hsv_blkim, cv2.COLOR_HSV2RGB)
    else:
        hsv_blkim = cv2.cvtColor(blank_im, cv2.COLOR_BGR2RGB)
        color_palette = A2D2_COLOR_PALETTE_SHORT
        for i in range(points.shape[0]):
            # cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)
            if color[i] < 0 or color[i] > 10:
                continue
            cv2.circle(hsv_blkim, (np.int32(points[i,1]), np.int32(points[i,0])), 2, color_palette[color[i]],-1)
        rgb_blkim = hsv_blkim
    
    # cv2.imwrite(w_path, rgb_blkim)
    # return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return rgb_blkim


def draw_registered_point(pc_ls, color_ls, vis=True, save=False):
    """
    Function to visualize register points
    Args:
        pc_ls: list of pc (np.array), which have been transformed in the same pose
        color_ls: list of color (ls), which has the same length of pc_ls
    Return:
        None
    """
    t_sample_ls = []
    for i in range(0, len(pc_ls)):
        source_temp = copy.deepcopy(pc_ls[i-1])
        # transform np.ndarray to PointCloud
        curr_pc = o3d.geometry.PointCloud()
        curr_pc.points = o3d.utility.Vector3dVector(source_temp[:, :3])
        # curr_pc = curr_pc.random_down_sample(0.04)
        curr_pc.estimate_normals()
        t_sample_ls.append(curr_pc)

    for i in range(len(t_sample_ls)):
        t_sample_ls[i].paint_uniform_color(color_ls[i])
    if save:
        for i in range(len(t_sample_ls)):
            o3d.io.write_point_cloud("mopa/samples/temp/{:05d}.pcd".format(i), t_sample_ls[i])
    if vis:
        o3d.visualization.draw_geometries(t_sample_ls,
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])


def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)

def depth_color(val, min_d=0, max_d=120):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val) # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 

def grep_depth_color(val, min_d=0, max_d=50):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val,) # max distance is 120m but usually not usual

    return (((max_d - val) / (max_d - min_d)) * (255))

def draw_range_image_depth(depth, save=False):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    # depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    # colors = []
    # for depth_val in depth.tolist():
    #     colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    grey_colors = grep_depth_color(depth)
    # ax5.imshow(np.full_like(img, 255))
    # plt.imshow(img)
    # plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    # plt.axis('off')

    # if show:
    #     plt.show()

    if save:
        from PIL import Image
        im = Image.fromarray(np.uint8(grey_colors), 'L')
        # save_img = grey_colors.astype(np.uint8)
        # save_img = np.stack((save_img[:,:,2], 
        #                      save_img[:,:,1],
        #                      save_img[:,:,0]), axis=2)
        print(grey_colors.shape)
        im.save('mopa/samples/grep_fig.jpg')


def draw_bird_eye_view(coords, full_scale=4096):
    plt.scatter(coords[:, 0], coords[:, 1], s=0.1)
    plt.xlim([0, full_scale])
    plt.ylim([0, full_scale])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# cuboid visualization function
def save_cuboid_centers_to_obj(cuboid_centers, extents, filename):
    with open(filename, 'w') as file:
        # Write vertex positions for cuboid centers and corners
        vertex_index = 1  # Starting index for vertices
        for center in cuboid_centers:
            x, y, z = center.tolist()
            extent_x, extent_y, extent_z = extents.tolist()
            vertices = [
                [x - extent_x, y - extent_y, z - extent_z],
                [x - extent_x, y - extent_y, z + extent_z],
                [x - extent_x, y + extent_y, z - extent_z],
                [x - extent_x, y + extent_y, z + extent_z],
                [x + extent_x, y - extent_y, z - extent_z],
                [x + extent_x, y - extent_y, z + extent_z],
                [x + extent_x, y + extent_y, z - extent_z],
                [x + extent_x, y + extent_y, z + extent_z]
            ]
            # Write vertex positions for cuboid centers and corners
            color = [1.0, 0.0, 0.0]
            for vertex in [center] + vertices:
                line = f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
                file.write(line)
                line = f"vc {color[0]} {color[1]} {color[2]}\n"
                file.write(line)
                vertex_index += 1

        # Write faces for each cuboid
        num_centers = len(cuboid_centers)
        for i in range(num_centers):
            base_index = i * 9  # Starting index for the cuboid's vertices
            line = f"f {base_index+1} {base_index+2} {base_index+4} {base_index+3}\n"
            file.write(line)
            line = f"f {base_index+5} {base_index+6} {base_index+8} {base_index+7}\n"
            file.write(line)
            line = f"f {base_index+1} {base_index+2} {base_index+6} {base_index+5}\n"
            file.write(line)
            line = f"f {base_index+3} {base_index+4} {base_index+8} {base_index+7}\n"
            file.write(line)
            line = f"f {base_index+1} {base_index+5} {base_index+7} {base_index+3}\n"
            file.write(line)
            line = f"f {base_index+2} {base_index+6} {base_index+8} {base_index+4}\n"
            file.write(line)
            

def debug_visualizer(
    pc: np.ndarray,
    save_pth: str,
    pc_color: np.ndarray = None,
) -> int:
    """Function for debug visualization

    Args:
        pc (np.ndarray): point cloud to be visualized (N, 3)
        pc_color (np.ndarray): RGB colors [0, 1] of each points (N, 3)
        save_pth (str): path to save the visualization

    Returns:
        int: simple return
    """
    device = o3d.core.Device("CPU:0")
    pc_o3d = o3d.t.geometry.PointCloud(device)
    pc_o3d.point['positions'] = o3d.core.Tensor(pc[:, :3], o3d.core.float32, device)
    if pc_color is not None:
        pc_o3d.point['colors'] = o3d.core.Tensor(pc_color, o3d.core.float32, device)
    o3d.t.io.write_point_cloud(save_pth, pc_o3d)
    
    return 0


def show_anns(sam_mask: np.ndarray, image: np.ndarray):
    """Simple visualization from SAM code

    Args:
        anns (dict): output mask generated by mask_generator
    """
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.zeros((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    for ann in np.unique(sam_mask):
        if ann == -100:
            continue
        m = sam_mask == ann
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def draw_point_image_depth(pc, image, img_indices, save_pth):
    dist = np.sqrt(pc[:,0] ** 2 + \
                  pc[:,1] ** 2 + \
                  pc[:,2] ** 2)

    c_ = depth_color(dist, dist.min(), dist.max())
    point_ori = print_projection_plt(img_indices, c_, image.shape)
    proj_img = cv2.addWeighted(image, 0.5, point_ori, 0.5, 1.0)
    cv2.imwrite(save_pth, proj_img)


def image_label_visualizer(
    labels_2d: np.ndarray,
    raw_image: np.ndarray,
    save_pth: str,
    filter: list = None
):
    """Function to visualize labeled image

    Args:
        labels_2d (np.ndarray): Pixel-wise labels (H, W)
        raw_image (np.ndarray): Input image array (3, H, W)
        save_pth (str): path to save visualization
    """

    base_color_map = [
        [245, 150, 100],  # car
        [180, 30, 80],  # truck
        [150, 60, 30],  # bike
        [30, 30, 255],  # person
        [255, 0, 255],  # road
        [255, 150, 255],  # parking
        [75, 0, 75],  # sidewalk
        [0, 200, 255],  # building
        [0, 175, 0],  # nature
        [255, 255, 50],  # other-objects
        [0, 0, 0],  # ignore
    ]
    if filter is not None:
        color_map = [base_color_map[-1]] * len(base_color_map)
        for i in filter:
            color_map[i] = base_color_map[i]
    else:
        color_map = base_color_map
    color_map = np.asarray([(c[2], c[1], c[0]) for c in color_map])
    logit_color = color_map[labels_2d].astype(np.uint8)
    logit_color = cv2.cvtColor(logit_color, cv2.COLOR_BGR2RGB)
    image = np.transpose((raw_image * 255), axes=[1,2,0]).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labeled_img = cv2.addWeighted(image, 0.6, logit_color, 0.4, 1.0)
    cv2.imwrite(save_pth, labeled_img)
    

def world_to_img(pc_array, intrin_mtx, img_size, return_idx=False):
    points_hcoords = np.concatenate([pc_array, np.ones([pc_array.shape[0], 1])], axis=1)
    img_points = (intrin_mtx @ points_hcoords.T).T
    img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)
    keep_index = select_points_in_frustum(img_points, 0, 0, *img_size)

    if not return_idx:
        return pc_array[keep_index], np.fliplr(img_points[keep_index])
    else:
        return pc_array[keep_index], np.fliplr(img_points[keep_index]), keep_index

    
def sample_visualizer(dataset: str = "waymo"):
    import os
    
    Ps = np.array([532.740352, 0, 640.0000, 0, 0, 532.740352, 380.0, 0, 0, 0, 1, 0])
    intrinc_mtx = Ps.reshape(3, 4)
    
    
    if dataset == "waymo":
        root_dir = "mopa/datasets/waymo/waymo_extracted/training"
        seq_dir = "segment-473735159277431842_630_095_650_095_with_camera_labels"
        scan_id = "000000"
        image_pth = os.path.join(root_dir, seq_dir, "camera_01/image/frame{}.png".format(scan_id))
        pcd_pth = os.path.join(root_dir, seq_dir, "camera_01/lidar/frame{}.pcd".format(scan_id))
        cp_lidar_pth = os.path.join(root_dir, seq_dir, "camera_01/cp_lidar/frame{}.npy".format(scan_id))
        gt_label_pth = os.path.join(root_dir, seq_dir, "camera_01/label/frame{}.npy".format(scan_id))
        
        image = Image.open(image_pth)
        points = np.asarray(o3d.io.read_point_cloud(pcd_pth).points)
        cp_lidar = np.load(cp_lidar_pth)
        labels = np.load(gt_label_pth).astype(np.int32)
        label_color_map = np.asarray(WAYMO_COLOR_PALETTE)
        label_colors = label_color_map[labels]
    elif dataset == "synthia":
        Ps = np.array([532.740352, 0, 640.0000, 0, 0, 532.740352, 380.0, 0, 0, 0, 1, 0])
        intrin_mtx = Ps.reshape(3, 4)
        
        root_dir = "mopa/datasets/synthia/sequence01"
        seq_dir = "SYNTHIA-SEQS-01-WINTER"
        scan_id = "000000"
        image_pth = os.path.join(root_dir, seq_dir, "RGB/Stereo_Left/Omni_F/{}.png".format(scan_id))
        pcd_pth = os.path.join(root_dir, seq_dir, "Lidar/Stereo_Left/Omni_F/{}.pcd".format(scan_id))
        gt_label_pth = os.path.join(root_dir, seq_dir, "Label/Stereo_Left/Omni_F/{}.npy".format(scan_id))
        
        image = Image.open(image_pth)
        points = np.asarray(o3d.io.read_point_cloud(pcd_pth).points)
        labels = np.load(gt_label_pth).astype(np.int32)
        label_color_map = np.asarray(WAYMO_COLOR_PALETTE)
        label_colors = label_color_map[labels]
        
        points, cp_lidar, keep_idx = world_to_img(points, intrin_mtx, image.size, return_idx=True)
        cp_lidar = np.fliplr(cp_lidar)
        label_colors = label_colors[keep_idx]
    else:
        raise AssertionError("Not implemented visualization")
    
    depth = np.sqrt(
        np.square(points[:, 0]) + \
        np.square(points[:, 1]) + \
        np.square(points[:, 2])
        )
    normalized_depth = ((depth - np.min(depth)) / (np.max(depth) - np.min(depth))) * 255
    normalized_depth = normalized_depth.astype(np.uint8)
    color_map = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    color_map = np.squeeze(color_map, axis=1)
    
    depth_img = np.zeros_like(image, dtype=np.uint8)
    for i in range(cp_lidar.shape[0]):
        color = color_map[i].tolist()
        cv2.circle(depth_img, (int(cp_lidar[i, 0]), int(cp_lidar[i, 1])),
                   3, color, -1)
        
    label_img = np.zeros_like(image, dtype=np.uint8)
    for i in range(cp_lidar.shape[0]):
        color = label_colors[i].tolist()
        cv2.circle(label_img, (int(cp_lidar[i, 0]), int(cp_lidar[i, 1])),
                   3, color, -1)
    
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    combined_image = cv2.addWeighted(image, 0.6, depth_img, 0.4, 1.0)
    labeled_image = cv2.addWeighted(image, 0.6, label_img, 0.4, 1.0)
    
    cv2.imwrite("mopa/samples/qe_figures/{}_raw_image.jpg".format(dataset), image)
    cv2.imwrite("mopa/samples/qe_figures/{}_heat_image.jpg".format(dataset), combined_image)
    cv2.imwrite("mopa/samples/qe_figures/{}_label_image.jpg".format(dataset), labeled_image)
    
if __name__ == "__main__":
    sample_visualizer("synthia")
    
    
    