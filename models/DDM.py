import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision.transforms import transforms
import numpy as np


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

class DDM(nn.Module):
    def __init__(self, num_blocks, filter, num_classes, dim):
        super(DDM, self).__init__()
        print('--Building DDM. ', end='')
        self.filter = filter
        self.in_planes = filter[0]
        self.block = PreActBlock

        self.cross_param = nn.Parameter(torch.ones(2,4,2), requires_grad=True)
        self.cross_param[:, :, 1].data.fill_(0)

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

        self.linear_exp_feat = nn.Sequential(
            nn.Dropout(),  # SFEW need 0.8
            nn.Linear(2048, dim),  #5888
            nn.PReLU()
        )

        self.linear_exp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(dim, num_classes),
            nn.PReLU()
        )

        self.linear_dist_feat = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, dim), #1472
            nn.PReLU()
        )

        # attention modules
        self.encoder_att = [0] * 2
        self.encoder_block_att = [0] * 2
        for i in range(2):
            self.encoder_att[i] = nn.ModuleList( [self._att_layer(filter[0])]) # same channel, same size, activation value (0,1)
            self.encoder_block_att[i] = nn.ModuleList([self._conv_layer([filter[0], filter[1]])]) # different channel, same size
        
        for t in range(2):
            for i in range(4): # num_block - 1
                self.encoder_att[t].append(self._att_layer(filter[i + 1]))

        for t in range(2):
            for i in range(4): # num_block - 1
                if i < 3: # num_block - 1 - 1
                    self.encoder_block_att[t].append(self._conv_layer([filter[i + 1], filter[i + 2]]))
                else:
                    self.encoder_block_att[t].append(self._conv_layer([filter[i + 1], filter[i + 1]]))
        self.encoder_att = nn.Sequential(*self.encoder_att)
        self.encoder_block_att = nn.Sequential(*self.encoder_block_att)

        self.ca_block = []
        for i in range(5):
            self.ca_block.append(self._ca_layer(channel=2*filter[i]))
        self.ca_block = nn.Sequential(*self.ca_block)

    def _conv_layer(self, channel): #1conv:channel[0]->chennel[1], same size
        conv_block = nn.Sequential(
            nn.BatchNorm2d(channel[0]),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[1]),
            nn.PReLU(),
        )
        return conv_block

    def _att_layer(self, channel):
        att_block = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )
        return att_block

    def _ca_layer(self, channel, reduction=16):
        # global average pooling: feature --> point
        # feature channel downscale and upscale --> channel weight
        ca_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        return ca_block

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

        # define attention block (num_attri, num_conv_block, num_step)
        atten_block = [0] * 2  # num_attri
        for i in range(2):
            atten_block[i] = [0] * 5    # num_conv_block
        for i in range(2):
            for j in range(5):
                atten_block[i][j] = [0] * 3

        for i in range(2): # attribute_num
            for j in range(5): # block_num
                if j == 0:
                    atten_block[i][j][0] = self.encoder_att[i][j](conv_layer[0])
                    atten_block[i][j][1] = (atten_block[i][j][0]) * conv_layer[j]
                    atten_block[i][j][2] = self.encoder_block_att[i][j](atten_block[i][j][1])
                    atten_block[i][j][2] = F.max_pool2d(atten_block[i][j][2], kernel_size=3, stride=2, padding=1)
                else:
                    conv_layer[j] = self.conv[j](conv_layer[j-1])

                    atten_block[i][j - 1][2] = self.apply_cross_stitch(
                            conv_layer[j], atten_block[i][j - 1][2], self.cross_param[i][j-1])
                    atten_block[i][j][0] = self.encoder_att[i][j](atten_block[i][j - 1][2])
                    atten_block[i][j][1] = (atten_block[i][j][0]) * conv_layer[j]
                    atten_block[i][j][2] = self.encoder_block_att[i][j](atten_block[i][j][1])
                    if j < 4: # block_num - 1
                        atten_block[i][j][2] = F.max_pool2d(atten_block[i][j][2], kernel_size=3, stride=2, padding=1)

        # print(atten_block[0][-1][-1].size())
        exp_feat = F.avg_pool2d(atten_block[0][-1][-1], 3)
        # print(exp_feat.shape)
        exp_feat = exp_feat.view(exp_feat.size(0), -1)
        exp_feat = self.linear_exp_feat(exp_feat)
        exp_pred = self.linear_exp(exp_feat)

        dist_feat = F.avg_pool2d(atten_block[1][-1][-1], 6)
        dist_feat = dist_feat.view(dist_feat.size(0), -1)
        dist_feat = self.linear_dist_feat(dist_feat)

        return exp_feat, exp_pred, dist_feat, (atten_block[1][0][-1], atten_block[1][1][-1], atten_block[1][2][-1], atten_block[1][3][-1], atten_block[1][4][-1])

    def apply_cross_stitch(self, input1, input2, cross_param):
        cross_param = F.softmax(cross_param, dim=0)
        # print(cross_param)
        # output1 = cross_param[0] * input1 + cross_param[1] * input2
        output2 = cross_param[0] * input2 + cross_param[1] * input1
        return output2