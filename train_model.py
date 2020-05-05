import os
import logging

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from data import iterator_factory
from train import metric
from train.model import model
from train.lr_scheduler import MultiFactorScheduler


def train_model(sym_net, model_prefix, dataset, input_conf,
                clip_length=16, train_frame_interval=2, val_frame_interval=2,
                resume_epoch=-1, batch_size=4, save_frequency=1,
                lr_base=0.01, lr_factor=0.1, lr_steps=[400000, 800000],
                end_epoch=1000, distributed=False, 
                pretrained_3d=None, fine_tune=False,
                load_from_frames=True, use_flow=False,
                **kwargs):

    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    # data iterator
    iter_seed = torch.initial_seed() \
                + (torch.distributed.get_rank() * 10 if distributed else 100) \
                + max(0, resume_epoch) * 100
    train_iter, eval_iter = iterator_factory.creat(name=dataset,
                                                   batch_size=batch_size,
                                                   clip_length=clip_length,
                                                   train_interval=train_frame_interval,
                                                   val_interval=val_frame_interval,
                                                   mean=input_conf['mean'],
                                                   std=input_conf['std'],
                                                   seed=iter_seed,
                                                   load_from_frames=load_from_frames,
                                                   use_flow=use_flow)
    # wapper (dynamic model)
    if use_flow:
        class LogNLLLoss(torch.nn.Module):
            def __init__(self):
                super(LogNLLLoss, self).__init__()
                self.loss = torch.nn.NLLLoss()

            def forward(self, output, target):
                output = torch.log(output)
                loss = self.loss(output, target)
                return loss
        # criterion = LogNLLLoss().cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    net = model(net=sym_net,
                criterion=criterion,
                model_prefix=model_prefix,
                step_callback_freq=50,
                save_checkpoint_freq=save_frequency,
                opt_batch_size=batch_size, # optional
                )
    net.net.cuda()

    # config optimization
    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    for name, param in net.net.named_parameters():
        if fine_tune:
            # if name.startswith('classifier'):
            if 'classifier' in name:
                param_new_layers.append(param)
            else:
                param_base_layers.append(param)
                name_base_layers.append(name)
        else:
            param_new_layers.append(param)

    if name_base_layers:
        out = "[\'" + '\', \''.join(name_base_layers) + "\']"
        logging.info("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers),
                     out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

    if distributed:
        net.net = torch.nn.parallel.DistributedDataParallel(net.net).cuda()
    else:
        net.net = torch.nn.DataParallel(net.net).cuda()

    optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': 0.2},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr=lr_base,
                                momentum=0.9,
                                weight_decay=0.0001,
                                nesterov=True)

    # load params from pretrained 3d network
    if pretrained_3d:
        if resume_epoch < 0:
            assert os.path.exists(pretrained_3d), "cannot locate: `{}'".format(pretrained_3d)
            logging.info("Initializer:: loading model states from: `{}'".format(pretrained_3d))
            checkpoint = torch.load(pretrained_3d)
            net.load_state(checkpoint['state_dict'], mode='ada')
        else:
            logging.info("Initializer:: skip loading model states from: `{}'"
                + ", since it's going to be overwrited by the resumed model".format(pretrained_3d))

    # resume training: model and optimizer
    if resume_epoch < 0:
        epoch_start = 0
        step_counter = 0
    else:
        net.load_checkpoint(epoch=resume_epoch, optimizer=optimizer)
        epoch_start = resume_epoch
        step_counter = epoch_start * train_iter.__len__()

    # set learning rate scheduler
    num_worker = 1
    lr_scheduler = MultiFactorScheduler(base_lr=lr_base,
                                        steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
                                        factor=lr_factor,
                                        step_counter=step_counter)
    # define evaluation metric
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),)
    # enable cudnn tune
    cudnn.benchmark = True

    net.fit(train_iter=train_iter,
            eval_iter=eval_iter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            epoch_start=epoch_start,
            epoch_end=end_epoch,)
