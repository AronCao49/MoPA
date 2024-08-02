import os
from yacs.config import CfgNode

def pspath_sort(
        cfg: CfgNode, 
        lidar_pth: str,
        pselab_dir: str,
        ) -> str:
    """
    Sorting function to output ps_label path

    Args:
        1. cfg:
            CfgNode that contains all args of testing.
        2. lidar_pth:
            The path to current evaluating lidar scan.
        3. pselab_dir:
            Directory to store pseudo-labels.
    Return:
        1. ps_label_pth:
            The path to pseudo label that fits the specific dataset format
    """
    if "SemanticKITTI".upper() in str(cfg.DATASET_TARGET.TYPE).upper():
        scen_dir, scan = lidar_pth.split('/')[-3], lidar_pth.split('/')[-1]
    elif "NuScenes".upper() in str(cfg.DATASET_TARGET.TYPE).upper():
        scen_dir, scan = lidar_pth.split('/')[-2], lidar_pth.split('/')[-1]
    else:
        raise AssertionError(
            "The required dataset is not supported yet: {}"\
                .format(cfg.DATASET_TARGET.TYPE)
            )
    pselab_name = scan.replace(".bin", ".npy")
    if not os.path.exists(os.path.join(pselab_dir, scen_dir)):
        os.makedirs(os.path.join(pselab_dir, scen_dir))
    ps_label_pth = os.path.join(pselab_dir, scen_dir, pselab_name)

    return ps_label_pth
