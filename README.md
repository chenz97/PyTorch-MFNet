# Action Retrieval on HMDB 51 

陈震 1901213532, 韩愉 1901213533, 李一博 1901213538

## Pre-requisites
- Linux
- Python 3
- PyTorch > 0.4.1
- Numpy
- Scikit-learn
- opencv-python

## Data Preparation

Download the HMDB-51 dataset. Then put the videos in dataset/HMDB51/raw/data. Then run the following commands:

```shell
cd dataset/HMDB51/scripts
python convert_videos.py
python extract_frames_cv2.py
```

If you would like to train with optical flow, please extract the optical flow for the dataset following https://github.com/feichtenhofer/gpu_flow, and put the extracted flow in dataset/HMDB51/raw/frames.

Download the pre-trained model on Kinetics at https://drive.google.com/file/d/1F7wvgZFZKEtoB284H5bBPaQZ7YiwrFQD/view, and put it in network/pretrained.

## Usage

### Training:

```shell
python train_hmdb51.py
```
Frequently used options:
- load-from-frames: if specified, use frames instead of videos as input. Strongly recommended.
- network: 'MFNet_3D' or 'DynImgNet'
- use-flow: if specified, optical flow is used as input. Only work with MFNet_3D
- dyn-mode: mode for dynamic image. 'dyn' or 'in_avg' or 'mid_dyn' or 'mid_avg' or 'in_concat'. Only work with DynImgNet 
- triplet-loss: if specified, use triplet loss. Only work with MFNet_3D

Example usage:

```shell
pyton train_hmdb51.py --load-from-frames
```



### Evaluate the trained model:

First, put the baseline_ep-0040.pth in exps/models/

```shell
# pre-compute and store gallery feature
python storage_feature.py --load-from-frames --task-name exps/models/baseline --load-epoch 40 --gpus 0 --split test
python storage_feature.py --load-from-frames --task-name exps/models/baseline --load-epoch 40 --gpus 0 --split others
cd test
# evaluate all
python evaluate_video_hmdb51_split1.py --load-from-frames --task-name ../exps/models/baseline --load-epoch 40 --gpus 0 --topN 10
python evaluate_video_hmdb51_split1.py --load-from-frames --task-name ../exps/models/baseline --load-epoch 40 --gpus 0 --topN 50
python evaluate_video_hmdb51_split1.py --load-from-frames --task-name ../exps/models/baseline --load-epoch 40 --gpus 0 --topN 200
```



## Citation

The original code is at https://github.com/cypw/PyTorch-MFNet

If you use the code/model in your work or find it is helpful, please cite the paper:
```
@inproceedings{chen2018multifiber,
  title={Multi-Fiber networks for Video Recognition},
  author={Chen, Yunpeng and Kalantidis, Yannis and Li, Jianshu and Yan, Shuicheng and Feng, Jiashi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2018}
}
```
