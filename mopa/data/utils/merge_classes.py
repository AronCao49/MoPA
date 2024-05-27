import numpy as np


train_label_name_mapping = {
        0: 'car',
        1: 'bicycle',
        2: 'motorcycle',
        3: 'truck',
        4: 'other-vehicle',
        5: 'person',
        6: 'bicyclist',
        7: 'motorcyclist',
        8: 'road',
        9: 'parking',
        10: 'sidewalk',
        11: 'other-ground',
        12: 'building',
        13: 'fence',
        14: 'vegetation',
        15: 'trunk',
        16: 'terrain',
        17: 'pole',
        18: 'traffic-sign'
    }

class_name_to_id = {v: k for k, v in train_label_name_mapping.items()}

# use those categories if merge_classes == True (common with A2D2)
categories = {
    'car': ['car'],
    'truck': ['truck'],
    'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist'],  # riders are labeled as bikes in Audi dataset
    'person': ['person'],
    'road': ['road'],
    'parking': ['parking'],
    'sidewalk': ['sidewalk'],
    'building': ['building'],
    'nature': ['vegetation', 'trunk', 'terrain'],
    'pole': ['pole'],
    'other-objects': ['fence', 'traffic-sign'],
}

# use those categories if merge_classes == True (common with A2D2)
categories_w = {
    'car': ['car', 'bus'],
    'truck': ['truck'],
    'bike': ['bicycle', 'motorcycle'],  # riders are labeled as bikes in Audi dataset
    'person': ['person'],
    'road': ['road'],
    'sidewalk': ['sidewalk'],
    'building': ['building', 'wall'],
    'nature': ['vegetation', 'terrain'],
    'pole': ['pole'],
    'trunk': ['trunk'],
    'traffic-sign': ['traffic sign'],
    'other-objects': ['fence',],
}

def merge_classes_kitti():
    highest_id = list(train_label_name_mapping.keys())[-1]
    label_mapping = 11 * np.ones(highest_id + 1, dtype=int)
    for cat_idx, cat_list in enumerate(categories.values()):
        for class_name in cat_list:
            label_mapping[class_name_to_id[class_name]] = cat_idx
    return label_mapping

def merge_classes_waymo():
    highest_id = list(train_label_name_mapping.keys())[-1]
    label_mapping = -1 * np.ones(highest_id + 1, dtype=int)
    for cat_idx, cat_list in enumerate(categories_w.values()):
        for class_name in cat_list:
            label_mapping[class_name_to_id[class_name]] = cat_idx
    return label_mapping