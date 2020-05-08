# python evaluate_video_hmdb51_split1.py --task-name ../exps/models/lr002_frame_cv2 --load-epoch 40 --gpus 2 --topN 50
import sys

sys.path.append("..")

import os
import time
import json
import logging
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sklearn import preprocessing
import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Evaluation)")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='HMDB51', choices=['HMDB51', 'UCF101', 'Kinetics'],
                    help="path to dataset")
parser.add_argument('--clip-length', default=16,
                    help="define the length of each input sample.")
parser.add_argument('--frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='../exps/models/PyTorch-MFNet-master',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="./eval-hmdb51-split1.log",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=int, default=1,
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='mfnet_3d',
                    choices=['mfnet_3d'],
                    help="chose the base network")
parser.add_argument('--use-flow', action='store_true')

# evaluation
parser.add_argument('--load-epoch', type=int, default=60,
                    help="resume trained model")
parser.add_argument('--batch-size', type=int, default=8,
                    help="batch size")
parser.add_argument('--topN', type=int, default=10,
                    help="topN")


def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    # fixed
    args.model_prefix = os.path.join(args.model_dir, args.task_name)
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
    feature_path = "./database/"
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


def get_top_N(Video_list, all_feature, N, V_feature):
    # print(all_feature.shape)
    # print(all_feature[:2,:])
    # print(V_feature.shape)
    # print(len(all_feature[0]))
    # print(V_feature)
    V_feature = preprocessing.normalize(V_feature.reshape(1, -1))
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
        # print(list_result[i])
        lre.append(list_result[i][1])
    return lre


def cal_AP(topN, real_label):
    true_label = real_label[:real_label.find("/")]
    # print(true_label)
    N = len(topN)
    count_i = 0.0
    number_i = 0.0
    s_AP = 0.0
    for name in topN:
        # print(name)
        number_i += 1.0
        tmpname = name[7:]
        tmpname1 = tmpname[tmpname.find("/") + 1:]
        pre_label = tmpname1[:tmpname1.find("/")]
        # print(pre_label)

        if pre_label == true_label:
            count_i += 1.0
            s_AP += count_i / number_i
    if count_i == 0:
        return 0
    return s_AP / count_i


if __name__ == '__main__':

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

    # data iterator:
    data_root = "../dataset/{}".format(args.dataset)
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
                           txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_test.txt'),
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

    net.net.eval()
    avg_score = {}
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    duplication = 1
    softmax = torch.nn.Softmax(dim=1)

    Video_list, feature_list = get_feature_dict()
    all_feature = np.array(feature_list)


    total_round = 1  # change this part accordingly if you do not want an inf loop
    for i_round in range(total_round):
        list_Ap = []
        i_batch = 0
        for data, target, video_subpath in eval_iter:
            # print(video_subpath)
            batch_start_time = time.time()
            feature = net.get_feature(data)
            feature = feature.detach().cpu().numpy()
            for i in range(len(video_subpath)):
                V_feature = feature[i]
                topN_re = get_top_N(Video_list, all_feature, args.topN, V_feature)
                tmp_AP = cal_AP(topN_re, video_subpath[i])
                print(video_subpath[i], str(tmp_AP))
                # print("              ")
                # print("              ")
                list_Ap.append(tmp_AP)
        sum_AP = 0.0
        for i in range(len(list_Ap)):
            sum_AP = sum_AP + list_Ap[i]
        MAP = sum_AP / len(list_Ap)
        print("MAP:", MAP)




