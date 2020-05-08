# python storage_feature.py --task-name exps/models/lr002_flow_real_fea_cat --load-epoch 60 --gpus 2  --use-flow --split test
import os
import logging
import cv2
import time
import json
import subprocess
import argparse
from joblib import delayed
from joblib import Parallel

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from network.symbol_builder import get_symbol
from train_model import train_model
import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol

import torchvision.models as models
# from torchsummary import summary

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Evaluation)")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='HMDB51', choices=['UCF101', 'Kinetics'],
                    help="path to dataset")
parser.add_argument('--clip-length', default=16,
                    help="define the length of each input sample.")
parser.add_argument('--frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='exps/models/PyTorch-MFNet-master',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="./eval-hmdb51.log",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=int, default=1,
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='mfnet_3d',
                    choices=['mfnet_3d'],
                    help="chose the base network")
parser.add_argument('--use-flow', action='store_true')
parser.add_argument('--dyn-mode', type=str, default='dyn')
# evaluation
parser.add_argument('--load-epoch', type=int, default=60,
                    help="resume trained model")
parser.add_argument('--batch-size', type=int, default=8,
                    help="batch size")
parser.add_argument('--split', type=str, default='test', choices=['test', 'others'])

##video_resize & extract_frame
##dst_folder="./query/frames"
'''
def exe_cmd(cmd):
    try:
        dst_file = cmd.split()[-1]
        if os.path.exists(dst_file):
            return "exist"
        cmd = cmd.replace('(', '\(').replace(')', '\)').replace('\'', '\\\'')
        output = subprocess.check_output(cmd, shell=True, 
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        logging.warning("failed: {}".format(cmd))
        # logging.warning("failed: {}: {}".format(cmd, err.output.decode("utf-8"))) # more details
        return False
    return output
'''

'''
def convert_video_wapper(src_videos,resize_videos,dst_videos):
    cmd_format = 'ffmpeg -y -i {} -c:v mpeg4 -filter:v "scale=min(iw\,(360*iw)/min(iw\,ih)):-1" -b:v 640k -an {}'
    commands=[]
    for src, dst in zip(src_videos, resize_videos):
        cmd = cmd_format.format(src, dst)
        commands.append(cmd)
    for i, cmd in enumerate(commands):
        exe_cmd(cmd=cmd)

    for src, dst in zip(resize_videos, dst_videos):
        print('dealing with ', src)
        vidcap = cv2.VideoCapture(src)
        cnt = 0
        success, image = vidcap.read()
        while success:
            cv2.imwrite(os.path.join(dst, '%05d.jpg' % cnt), image)
            success, image = vidcap.read()
            cnt += 1
        print(' - Done. Processed {} frames'.format(cnt))
'''

'''
def extract_query_frame():
    src_folder="./query/data"
    resize_folder="./query/videos"
    dst_folder='./query/frames'
    video_names=os.listdir(src_folder)
    for vid_name in video_names:
        folder = os.path.join(dst_folder, vid_name[:-4])
        if not os.path.exists(folder):
            os.makedirs(folder)
    src_videos = [os.path.join(src_folder, vid_name) for vid_name in video_names]
    resize_videos=[os.path.join(resize_folder,vid_name) for vid_name in video_names]
    dst_videos = [os.path.join(dst_folder, vid_name[:-4]) for vid_name in video_names]
    convert_video_wapper(src_videos=src_videos,resize_videos=resize_videos,dst_videos=dst_videos)
'''


def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    # fixed
    args.model_prefix = os.path.join(args.model_dir, args.task_name)
    # 设置model的位置
    return args


def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./" + os.path.dirname(log_file)):
            os.makedirs("./" + os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


if __name__ == '__main__':
    # extract_query_frame()
    # set args
    args = parser.parse_args()
    args = autofill(args)

    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
    logging.info("Start evaluation with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)  # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)
    # number_class=51

    # creat model
    sym_net, input_config = get_symbol(name=args.network, use_flow=args.use_flow, **dataset_cfg)

    # network
    if torch.cuda.is_available():
        cudnn.benchmark = True
        sym_net = torch.nn.DataParallel(sym_net).cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        sym_net = torch.nn.DataParallel(sym_net)
        criterion = torch.nn.CrossEntropyLoss()
    net = static_model(net=sym_net,
                       criterion=criterion,
                       model_prefix=args.model_prefix)
    net.load_checkpoint(epoch=args.load_epoch)

    data_root = "./dataset/{}".format(args.dataset)
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    val_sampler = sampler.RandomSampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         speed=[1.0, 1.0])

    if args.use_flow:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensorMixed(dim1=3, dim2=2, t_channel=args.clip_length*3),
            normalize,
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    val_loader = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),  # change this part accordingly
                           frame_prefix=os.path.join(data_root, 'raw', 'frames'),
                           txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_{}.txt'.format(args.split)),
                           # change this part accordingly
                           sampler=val_sampler,
                           force_color=True,
                           use_flow=args.use_flow,
                           flow_prefix=os.path.join(data_root, 'raw', 'flow'),
                           video_transform=val_transform,
                           name='test',
                           return_item_subpath=True
                           )

    eval_iter = torch.utils.data.DataLoader(val_loader,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,  # change this part accordingly
                                            pin_memory=True)

    # main loop
    net.net.eval()
    avg_score = {}
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    duplication = 1
    softmax = torch.nn.Softmax(dim=1)

    total_round = 1  # change this part accordingly if you do not want an inf loop
    for i_round in range(total_round):
        i_batch = 0
        for data, target, video_subpath in eval_iter:
            batch_start_time = time.time()
            feature = net.get_feature(data)
            feature = feature.detach().cpu().numpy()
            for i in range(len(video_subpath)):
                name = video_subpath[i]
                folder_name = "test/database/" + name[:name.find('/')]
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                save_path = "test/database/" + name[:-4] + ".npy"
                np.save(save_path, feature[i])

