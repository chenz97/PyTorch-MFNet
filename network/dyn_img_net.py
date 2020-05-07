import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['DynamicImgNet', 'dynimgnet18', 'dynimgnet34', 'dynimgnet50',
           'dynimgnet101', 'dynimgnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1,
                     bias=False)  # with padding = 1, 3*3 conv won't change feature map size with stride = 1 (default setting)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class RankPool(nn.Module):
    """Implements the rank pooling operator for dynamic image
    """
    def __init__(self, sample_duration):
        super(RankPool, self).__init__()
        self.sample_duration = sample_duration
        self.h_lst = [0, ]
        self._harmonic_num_lst(self.sample_duration)
        self.alpha_vec = torch.zeros(self.sample_duration)
        for i in range(0, self.sample_duration):
            self.alpha_vec[i] = self._cal_alpha(i + 1)

    def _harmonic_num_lst(self, T):
        h_t = 0.0
        for i in range(1, T + 1):
            h_t += 1.0 / i
            self.h_lst.append(h_t)

    def _cal_alpha(self, t):
        return 2 * (self.sample_duration - t + 1) \
               - (self.sample_duration + 1) * (self.h_lst[self.sample_duration - 1] - self.h_lst[t - 1])

    def forward(self, x):
        weighed_vec = self.alpha_vec.reshape([1, 1, self.alpha_vec.size(0), 1, 1]).expand_as(x).cuda() * x
        return weighed_vec.sum(dim=2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    '''
        make channels change from inplanes to planes * expansion
        adjust feature map size accroding to the stride specified
            1 * 1 conv: used to change channel numbers from Cin (inplanes) to planes
            3 * 3 conv: conv (with padding which help maintain feature map size), but with stride which may reduce feature map size. channel is maintained
            1 * 1 conv: used to change channel numbers from planes to planes * expansion
            shortcut: downsample the input according the given downsample parameter
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # out_channel = planes * expansion
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)  # stride is only used here
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(
                x)  # the residual connection is consistent with the output, not necessarily the original input

        out += residual  # planes * expansion = downsample(inplanes)
        out = self.relu(out)

        return out


class DynamicImgNet(nn.Module):
    '''
        the parameters only care about the channel, without considering the feature map size
        feature map size is controlled by stride (and conv padding, which is usually set to make feature map size unchanged)
        the pooling before fc layer makes feature map be 1*1
    '''

    def __init__(self, block, layers, sample_duration=16, num_classes=1000, pretrained='', dyn_mode='dyn'):
        super(DynamicImgNet, self).__init__()
        self.inplanes = 64
        self.sample_duration = sample_duration
        self.mode = dyn_mode
        self.rankpool = RankPool(sample_duration)
        if dyn_mode == 'in_concat':
            self.conv1 = nn.Conv2d(3 * sample_duration, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # QUESTION: when to use stride > 1 ?
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # para: output_size, channel is maintained
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.score_fusion

        if pretrained:
            self.__load_fea_weights(torch.load(pretrained))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def __load_fea_weights(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or name.startswith('fc'):
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if param.size() != own_state[name].size():
                # inflate
                param = torch.repeat_interleave(param, int(own_state[name].size(1) / param.size(1)), dim=1) / \
                        (own_state[name].size(1) / param.size(1))
            own_state[name].copy_(param)

    def _make_layer(self, block, planes, blocks, stride=1):  # stride change the feature map size in the BottleNeck
        '''
            construct layer1/2/3/4 by stacking $blocks bottlenecks, each of which has a shortcut connection
                bottleneck:
                    1 * 1 conv, used to reduce channel number
                        for the first bottleneck, the input channel is layer (i-1)'s output channel, the output channel is $planes
                        for the following bottlenecks, the input channel is the current layer's output channel
                    3 * 3 conv, possibily (expect for layer1) with stride = 2 to half feature map size, without modifying channel number
                    1 * 1 conv, used to add channel number to the specified $planes * $expansion

                    shortcut: connect input to the ouput of the bottleneck, possibly using downsample
                    downsample: a 1 * 1 conv whose input channel is layer (i-1)'s output channel, the output channel is current layer's output channel, possibily with stride=2
                        for the first bottleneck, downsample is used
                        for the following bottleneck, downsample is not used

        '''

        downsample = None
        # downsample acts on the input layer for the residual connection, making the residual consistent with the output, both in feature map size and channel
        # when the stride is not 1, output feature map size would change
        # when inplanes is not equal to planes * expansion, the output feature map would change
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # repeat the structure $blocks times
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes))  # downsample is not needed since stride is already conducted and inplane is equal to planes * expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # CAUTION: Data is required to be in the format of (N, C, T, H, W)
        if self.mode == 'dyn':
            x = self.rankpool(x)
        elif self.mode == 'in_avg':
            x = x.mean(dim=2)
        elif self.mode == 'random':
            idx = int(torch.randint(0, self.sample_duration, (1,)).item())
            x = x[:, :, idx, :, :]
        elif self.mode == 'fea_avg' or self.mode == 'mid_avg'\
                or self.mode == 'fea_dyn' or self.mode == 'mid_dyn' or self.mode == 'in_concat':
            N = x.size(0)
            T = x.size(2)
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
            if self.mode == 'in_concat':
                x = x.view(N, T * x.size(2), x.size(3), x.size(4))
            else:
                x = x.view(N*T, x.size(2), x.size(3), x.size(4))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.mode == 'mid_dyn' or self.mode == 'mid_avg':
            x = x.view(N, T, x.size(1), x.size(2), x.size(3))
            x = x.permute(0, 2, 1, 3, 4)
            if self.mode == 'mid_dyn':
                x = self.rankpool(x)
            else:
                x = x.mean(dim=2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.mode == 'fea_avg':  # TODO: fea_dyn
            x = x.view(N, T, x.size(1), x.size(2), x.size(3))
            x = x.permute(0, 2, 1, 3, 4)
            x = self.avgpool3d(x)
        else:
            x = self.avgpool(x)  # make the feature map be 1*1

        x = x.view(x.size(0), -1)  # squeeze the unnecessary dims
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        parameters = []
        for v in model.parameters():
            parameters.append({'params': v})
        return parameters

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class TwoStreamDynImgNet(nn.Module):
    def __init__(self, block, layers, sample_duration=16, num_classes=1000, pretrained='', dyn_mode='dyn'):
        super(TwoStreamDynImgNet, self).__init__()
        self.rgb_stream = DynamicImgNet(block, layers, sample_duration, num_classes, pretrained, dyn_mode='mid_avg')
        self.dyn_stream = DynamicImgNet(block, layers, sample_duration, num_classes, pretrained, dyn_mode='mid_dyn')

    def forward(self, x):
        pred1 = self.rgb_stream(x)
        pred2 = self.dyn_stream(x)
        fusion_pred = (pred1 + pred2) / 2
        return fusion_pred

def dynimgnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DynamicImgNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def dynimgnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DynamicImgNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def dynimgnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DynamicImgNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def two_stream_dynimgnet50(**kwargs):
    model = TwoStreamDynImgNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def dynimgnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DynamicImgNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def dynimgnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DynamicImgNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model