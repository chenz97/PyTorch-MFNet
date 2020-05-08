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

from sklearn import preprocessing
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
from torchsummary import summary

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
# evaluation
parser.add_argument('--load-epoch', type=int, default=60,
                    help="resume trained model")
parser.add_argument('--batch-size', type=int, default=1,
                    help="batch size")


##video_resize & extract_frame
##dst_folder="./query/frames"


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


def convert_video_wapper(src_videos, resize_videos, dst_videos):
    cmd_format = 'ffmpeg -y -i {} -c:v mpeg4 -filter:v "scale=min(iw\,(360*iw)/min(iw\,ih)):-1" -b:v 640k -an {}'
    commands = []
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


def extract_query_frame():
    src_folder = "./query/data"
    resize_folder = "./query/videos"
    dst_folder = './query/frames'
    video_names = os.listdir(src_folder)
    for vid_name in video_names:
        folder = os.path.join(dst_folder, vid_name[:-4])
        if not os.path.exists(folder):
            os.makedirs(folder)
    src_videos = [os.path.join(src_folder, vid_name) for vid_name in video_names]
    resize_videos = [os.path.join(resize_folder, vid_name) for vid_name in video_names]
    dst_videos = [os.path.join(dst_folder, vid_name[:-4]) for vid_name in video_names]
    convert_video_wapper(src_videos=src_videos, resize_videos=resize_videos, dst_videos=dst_videos)


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


def get_feature_dict():
    # feature_dict={}
    feature_path = "./test/database/"
    dirs = os.listdir(feature_path)
    Video_list = []
    feature_list = []
    for f_dir in dirs:
        v_path = feature_path + f_dir
        f_names = os.listdir(v_path)
        for f_name in f_names:
            tmp_f = np.load(v_path + "/" + f_name)
            tmp_f = tmp_f.reshape(1, -1)
            # print(tmp_f[:3])
            tmp_f = preprocessing.normalize(tmp_f)
            # print(tmp_f[:3])
            tmp_f = np.squeeze(tmp_f)
            Video_list.append(v_path + "/" + f_name)
            feature_list.append(tmp_f)
            # feature_dict[v_path+"/"+f_name]=tmp_f
    # print(feature_list[:2,:])
    return Video_list, feature_list


def take_key(elem):
    return elem[0]


Video_list, feature_list = get_feature_dict()


def get_top_N(N, V_feature):
    all_feature = np.array(feature_list)
    V_feature = V_feature.reshape(1, -1)
    V_feature = preprocessing.normalize(V_feature)
    V_feature = np.squeeze(V_feature)
    list_result = []
    dis_all = np.sum(np.square(all_feature - V_feature), axis=1)
    for i in range(dis_all.shape[0]):
        list_result.append((dis_all[i], Video_list[i]))
    # for (key,value) in f_dict.items():
    # value=preprocessing.normalize(value.reshape(1,-1))
    # dis=np.sqrt(np.sum(np.square(value-V_feature)))
    # list_result.append((dis,key))
    list_result.sort(key=take_key)
    lre = []
    for i in range(N):
        print(list_result[i])
        lre.append(list_result[i][1])
    return lre


if __name__ == '__main__':
    extract_query_frame()
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
    sym_net, input_config = get_symbol(name=args.network, **dataset_cfg)

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

    data_root = "./query/"
    query_names = os.listdir(data_root + "videos")
    txt_path = "./query/list_cvt/search.txt"
    if os.path.exists(txt_path):
        os.remove(txt_path)
    with open(txt_path, "w")as f:
        for i in range(len(query_names)):
            f.write(str(i) + "\t" + "0" + "\t" + query_names[i] + "\n")

    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    val_sampler = sampler.RandomSampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         speed=[1.0, 1.0])
    val_loader = VideoIter(video_prefix=os.path.join(data_root, 'videos'),
                           frame_prefix=os.path.join(data_root, 'frames'),
                           txt_list=os.path.join(data_root, 'list_cvt', 'search.txt'),
                           sampler=val_sampler,
                           force_color=True,
                           video_transform=transforms.Compose([
                               transforms.Resize((256, 256)),
                               transforms.CenterCrop((224, 224)),
                               transforms.ToTensor(),
                               normalize,
                           ]),
                           name='test',
                           return_item_subpath=True
                           )
    eval_iter = torch.utils.data.DataLoader(val_loader,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=1,  # change this part accordingly
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
            feature = feature.detach().cpu().numpy()[0]
            get_top_N(20, feature)



