"""
Author: Yunpeng Chen
"""
import torch
import numpy as np

from .image_transforms import Compose, \
                              Transform, \
                              Normalize, \
                              Resize, \
                              RandomScale, \
                              CenterCrop, \
                              RandomCrop, \
                              RandomHorizontalFlip, \
                              RandomRGB, \
                              RandomHLS


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x (T x C)) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            H, W, _ = clips.shape
            # handle numpy array
            clips = torch.from_numpy(clips.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
            # backward compatibility
            return clips.float() / 255.0

class ToTensorMixed(Transform):
    """Converts a numpy.ndarray (H x W x (T x C1 + T x C2)) in the range
    [0, 255] to a torch.FloatTensor of shape (C1+C2 x T x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim1=3, dim2=2, t_channel=48):
        self.dim1 = dim1
        self.dim2 = dim2
        self.t_channel = t_channel

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            H, W, _ = clips.shape
            # handle numpy array
            clip1 = torch.from_numpy(clips[:, :, :self.t_channel].reshape((H, W, -1, self.dim1)).transpose((3, 2, 0, 1)))
            clip2 = torch.from_numpy(clips[:, :, self.t_channel:].reshape((H, W, -1, self.dim2)).transpose((3, 2, 0, 1)))
            clips = torch.cat((clip1, clip2), dim=0)
            # backward compatibility
            return clips.float() / 255.0