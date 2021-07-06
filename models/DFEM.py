import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision.transforms import transforms
import numpy as np

import pickle
import socket
hostname = socket.gethostname()



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

def weight_init(self, m):
        if isinstance(m,nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0,math.sqrt(2./n))
        elif isinstance(m,nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            param_shape = layer.weight.shape
            layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)) 

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1  

    def __init__(self, in_planes, planes, stride=1):  #inplane:input channels  plane:output channels
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.PRelu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.PRelu2 = nn.PReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
    def forward(self, x):
        out = self.PRelu1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.PRelu2(self.bn2(out)))
        out += shortcut
        return out


class DFEM(nn.Module):
    def __init__(self, num_blocks, filter, num_classes, dim):
        super(DFEM, self).__init__()
        print('--Building DFEM. ', end='')
        self.filter = filter
        self.in_planes = filter[0]
        self.block = PreActBlock

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, filter[0], kernel_size=3, stride=1, padding=1, bias=False), # 48
            nn.BatchNorm2d(filter[0]),
            nn.PReLU()
        )

        self.conv = []
        self.conv.append(self.layer0)
        for i in range(4):
            self.conv.append(self._make_layer(self.block, filter[i+1], num_blocks[i], stride=2))
        self.conv = nn.Sequential(*self.conv)

        self.fc0 = nn.Sequential(
            nn.Linear(2048, dim),
            nn.PReLU()
        )

        # self.fc_sub = nn.Sequential(
        #     nn.Linear(dim, 337),
        #     nn.PReLU()
        # )
        # self.fc_exp = nn.Sequential(
        #     nn.Linear(dim, num_classes),
        #     nn.PReLU()
        # )
        # self.fc_pose = nn.Sequential(
        #     nn.Linear(dim, 15),
        #     nn.PReLU()
        # )
        # self.fc_illu = nn.Sequential(
        #     nn.Linear(dim, 20),
        #     nn.PReLU()
        # )


        # self.fc_gen = nn.Sequential(
        #     nn.Linear(dim, 3),
        #     nn.PReLU()
        # )
        # self.fc_race = nn.Sequential(
        #     nn.Linear(dim, 3),
        #     nn.PReLU()
        # )
        # self.fc_age = nn.Sequential(
        #     nn.Linear(dim, 5),
        #     nn.PReLU()
        # )


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)


    def forward(self, x):
        # define global shared network
        conv_layer = [0] * 5
        conv_layer[0] = self.conv[0](x)
        conv_layer[1] = self.conv[1](conv_layer[0])
        conv_layer[2] = self.conv[2](conv_layer[1])
        conv_layer[3] = self.conv[3](conv_layer[2])
        conv_layer[4] = self.conv[4](conv_layer[3])
        feat = F.avg_pool2d(conv_layer[-1], 3)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc0(feat)

        # sub = self.fc_sub(feat)
        # exp = self.fc_exp(feat)
        # pose = self.fc_pose(feat)
        # illu = self.fc_illu(feat)

        # gen = self.fc_gen(feat)
        # race = self.fc_race(feat)
        # age = self.fc_age(feat)

        return feat, [conv_layer[1], conv_layer[2], conv_layer[3], conv_layer[4], conv_layer[4]]# [sub, exp, pose, illu] [exp, gen, race, age]