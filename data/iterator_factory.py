import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter

def get_hmdb51(data_root='./dataset/HMDB51',
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=0,
               load_from_frames=False,
               use_flow=False,
               **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    if use_flow:
        train_transform = transforms.Compose([
            transforms.RandomScale(make_square=True,
                                   aspect_ratio=[0.8, 1. / 0.8],
                                   slen=[224, 288]),
            transforms.RandomCrop((224, 224)),  # insert a resize if needed
            transforms.RandomHorizontalFlip(),
            transforms.RandomHLS(vars=[15, 35, 25], t_channels=clip_length*3),
            transforms.ToTensorMixed(dim1=3, dim2=2, t_channel=clip_length*3),
            normalize,
            ],
            aug_seed=(seed + 1))
    else:
        train_transform = transforms.Compose([
            transforms.RandomScale(make_square=True,
                                   aspect_ratio=[0.8, 1. / 0.8],
                                   slen=[224, 288]),
            transforms.RandomCrop((224, 224)),  # insert a resize if needed
            transforms.RandomHorizontalFlip(),
            transforms.RandomHLS(vars=[15, 35, 25], t_channels=clip_length*3),
            transforms.ToTensor(),
            normalize,
            ],
            aug_seed=(seed + 1))
    # NOTE: equivalent to Dataset, returns a sequence of frames
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data-x360'),
                      frame_prefix=os.path.join(data_root, 'raw', 'frames'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_train.txt'),
                      sampler=train_sampler,
                      flow_prefix=os.path.join(data_root, 'raw', 'flow'),
                      load_from_frames=load_from_frames,
                      use_flow=use_flow,
                      force_color=True,
                      video_transform=train_transform,
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    if use_flow:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensorMixed(dim1=3, dim2=2, t_channel=clip_length*3),
            normalize,
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data-x360'),
                      frame_prefix=os.path.join(data_root, 'raw', 'frames'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_test.txt'),
                      sampler=val_sampler,
                      flow_prefix=os.path.join(data_root, 'raw', 'flow'),
                      load_from_frames=load_from_frames,
                      use_flow=use_flow,
                      force_color=True,
                      video_transform=val_transform,
                      name='test',
                      )

    return (train, val)

def get_ucf101(data_root='./dataset/UCF101',
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=0,
               **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'trainlist01.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'testlist01.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )

    return (train, val)


def get_kinetics(data_root='./dataset/Kinetics',
                 clip_length=8,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=0,
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'train_avi-x256'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_train_w-missed-v1_avi.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val_avi-x256'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_w-missed-v1_avi.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )
    return (train, val)



def creat(name, batch_size, num_workers=16, **kwargs):

    if name.upper() == 'UCF101':
        train, val = get_ucf101(**kwargs)
    elif name.upper() == 'HMDB51':
        train, val = get_hmdb51(**kwargs)
    elif name.upper() == 'KINETICS':
        train, val = get_kinetics(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))


    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val,
        # batch_size=2*torch.cuda.device_count(), shuffle=False,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    return (train_loader, val_loader)