# Dataset Preparation
Below we provide how to prepare the dataset for MoPA, including:
* [A2D2](#1-a2d2)
* [NuScenes](#2-nuscenes)
* [SemanticKITTI](#3-semantickitti)
* [Rare Objects (ROs) from the wild](#4-rare-objects-ros-from-the-wild)
* [Segment Anything Model (SAM) masks](#5-segment-anything-model-sam-masks)

Before getting started, we recommand you to create a folder  ```mopa/datasets``` to store the soft links to each datasets for better data arrangement.

## 1. A2D2
Our preprocessing procedures of A2D2 are identical to [xMUDA](https://github.com/valeoai/xmuda). First download the raw A2D2 dataset from [their official website](https://a2d2.audi/a2d2/en.html). Then, create a soft link ```ln -sfn /path/to/raw/a2d2 mopa/datasets/a2d2```, and subsequently run the following preprocessing command:
```bash
$ python mopa/data/a2d2/preprocess.py
```

## 2. NuScenes
Similar to A2D2, our preprocessing on NuScenes is the same as [xMUDA](https://github.com/valeoai/xmuda), except that we add additional ground extraction for VGI. After download the NuScenes dataset from their [website](https://www.nuscenes.org/nuscenes), create a soft link ```ln -sfn /path/to/raw/nuscenes mopa/datasets/nuscenes```, and then run the following command:
```bash
$ python mopa/data/nuscenes/preprocess.py
```

## 3. SemanticKITTI
Since SemanticKITTI is larger compared to others, we employ a scan-by-scan loading strategy in our repo. After following the downloading instruction of [xMUDA](https://github.com/valeoai/xmuda), create a soft link ```ln -sfn /path/to/raw/semantickitti mopa/datasets/semantic_kitti```, and then run the following command:
```bash
$ python mopa/data/semantic_kitti/preprocess.py
```

## 4. Rare Objects (ROs) from the wild
As mentioned in our paper, MoPA mainly leverage labeled instance from the Waymo dataset to imporve segmentation on ROs. To extract those labeled instance, first download the raw dataset from [the official website](https://waymo.com/open/) of Waymo dataset and decompress them in the same folder (you can download the training split only). You now should have a dataset folder organized as follows:

📦Waymo <br> 
┣ 📂training <br>
┃ ┣ 📜segment-xxxxxxx_with_camera_labels.tfrecord   
┃ ┣ 📜segment-yyyyyyy_with_camera_labels.tfrecord  <br>
┗ ┗ 📜...  <br>

We recommand you to link the raw Waymo data folder to ```mopa/datasets/waymo``` under this repo. Subsequently, first create a isolated conda environment outside the docker container and install the requirements. 
```bash
$ conda create -n waymo_extract python=3.8
$ conda activate waymo_extract && pip install -r mopa/data/waymo/requirements.txt
```
Then, use the following command to extract frame-wise data from the *.tfrecord:
```bash
$ python mopa/data/waymo/data_extractor.py --raw_waymo_dir=/path/to/raw/waymo/dir
```
The frame-wise data would be automatically store in sub-folder named ```waymo_extracted``` under the raw Waymo directory, organized as follows:

📦Waymo <br> 
┣ 📂training <br>
┣ 📂waymo_extracted <br>
┃ ┣ 📂training <br>
┃ ┃ ┣ 📂segment-xxxxxxx_with_camera_labels <br>
┃ ┃ ┃ ┣ 📂lidar <br>
┃ ┃ ┃ ┃ ┣ 📜frame000000.pcd <br>
┃ ┃ ┃ ┃ ┗ 📜... <br>
┃ ┃ ┃ ┣ 📂label <br>
┃ ┃ ┃ ┃ ┣ 📜frame000000.label <br>
┃ ┃ ┃ ┃ ┗ 📜... <br>
┃ ┃ ┃ ┣ 📂bin <br>
┃ ┃ ┃ ┃ ┣ 📜frame000000.bin <br>
┃ ┃ ┃ ┗ ┗ 📜... <br>
┃ ┃ ┣ 📂segment-yyyyyyy_with_camera_labels <br>
┗ ┗ ┗ 📂...

Finally, attach to the docker container and use the following command to extract ROs from frame-wise data by DBSCAN:
```bash
$ python mopa/data/waymo/obj_point_extract.py \
$       --waymo_dir=/path/to/waymo/extracted/dir \
$       --obj_save_dir=/path/to/waymo/ROs/dir
```
You can try-out different extraction arguments in ```obj_point_extract.py```. By default, the extracted ROs would be stored under Waymo data dir as:

📦Waymo <br> 
┣ 📂training <br>
┣ 📂waymo_extracted <br>
┃ ┣ 📂training <br>
┃ ┣ 📂objects <br>
┃ ┃ ┣ 📂bicycle <br>
┃ ┃ ┃ ┣ 📜00001.bin <br>
┃ ┃ ┃ ┗ 📜... <br>
┃ ┃ ┣ 📂motorcycle <br>
┗ ┗ ┗ 📂pedestrian <br>

## 5. Segment Anything Model (SAM) masks
To utilize SAM masks for dense 2D supervision signals, first download the pretrained SAM model from [their repo](https://github.com/facebookresearch/segment-anything) (we use ```vit-h``` for MoPA in our paper). Then, use the following command to generate SAM mask for NuScenes and SemanticKITTI:
```bash
$ python mopa/data/sam_refine.py --model_type vit-h --sam_ckpt_pth /path/to/sam/checkpoint
```
The generated SAM mask are stored in the sub-directory ```img_mask``` for each dataset. 


