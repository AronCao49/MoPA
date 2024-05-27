import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from mopa.data.utils.refine_pseudo_labels import refine_pseudo_labels

class_name = [
        'car',
        'truck',
        'bike',
        'person',
        'road',
        'parking',
        'sidewalk',
        'building',
        'nature',
        'other-objects',
        ]


def check_pl_dstr(
    pl_dir: str,
    num_classes: int,
    class_filter: bool = True,
    gt_labels: bool = False
):
    """Simple function to check class-wise distribution of pslabels

    Args:
        pl_dir (str): Path to the pseudo-label directory (only for kitti for now)
        num_classes (int): Total number of classes.
        class_filter (bool, optional): Whether using class-wise filtering. 
            Defaults to True.
        gt_labels (bool, optional): Whether checking gt labels. Defaults to False.
    """
    sample_num_2d = np.zeros(num_classes)
    sample_num_3d = np.zeros(num_classes)
    
    for seq_dir in sorted(os.listdir(pl_dir)):
        curr_seq_dir = os.path.join(pl_dir, seq_dir)
        for ps_label_file in tqdm(os.listdir(curr_seq_dir)):
            if ".npy" not in ps_label_file:
                continue
            
            # Load ps_labels
            ps_data = np.load(
                os.path.join(curr_seq_dir, ps_label_file), allow_pickle=True
                ).tolist()
            probs_2d = ps_data['probs_2d']
            probs_3d = ps_data['probs_3d']
            ps_label_2d = ps_data['pseudo_label_2d']
            ps_label_3d = ps_data['pseudo_label_3d']
            
            # Refine labels if needed
            if class_filter:
                ps_label_2d = refine_pseudo_labels(
                    probs_2d, ps_label_2d.astype(np.int32)
                )
                ps_label_3d = refine_pseudo_labels(
                    probs_3d, ps_label_3d.astype(np.int32)
                )
            
            curr_portation_2d = np.zeros(num_classes)
            curr_portation_3d = np.zeros(num_classes)
            # Calculate class-wise portation (2D)
            for label in np.unique(ps_label_2d):
                if label < 0:
                    continue
                curr_portation_2d[int(label)] = \
                    np.count_nonzero(ps_label_2d == label) / ps_label_2d.shape[0]
            sample_num_2d += curr_portation_2d
            # Calculate class-wise portation (3D)
            for label in np.unique(ps_label_3d):
                if label < 0:
                    continue
                curr_portation_3d[int(label)] = \
                    np.count_nonzero(ps_label_3d == label) / ps_label_3d.shape[0]
            sample_num_3d += curr_portation_3d
    
    print("2D Class-wise orrurancy: {}".format(sample_num_2d.tolist()))
    print("3D Class-wise orrurancy: {}".format(sample_num_3d.tolist()))
    
    # Plot and save
    plt.figure(0)
    sns.barplot(x=class_name, y=sample_num_2d)
    plt.title("2D PsLabel Class-wise Occurancy")
    plt.xlabel("Class Names")
    plt.ylabel("Occurancies")
    plt.savefig("mopa/samples/pslabel_occu_2d.png")
    plt.close()
    
    plt.figure(1)
    sns.barplot(x=class_name, y=sample_num_3d)
    plt.title("3D PsLabel Class-wise Occurancy")
    plt.xlabel("Class Names")
    plt.ylabel("Occurancies")
    plt.savefig("mopa/samples/pslabel_occu_3d.png")
    plt.close()
    
if __name__ == "__main__":
    check_pl_dstr(
        "mopa/datasets/semantic_kitti/ps_label/1226_ps_label",
        10
    )
    