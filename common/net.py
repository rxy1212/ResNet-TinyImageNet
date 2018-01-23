'''
@file: net.py
@version: v1.0
@date: 2018-01-18
@author: ruanxiaoyi
@brief: Design the network
@remark: {when} {email} {do what}
'''

import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class ResNet(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet, self).__init__()
        self.toplayer = nn.Sequential(
            BasicConv2d(3, 64, 3, padding=1)
        )
        self.block1 = nn.Sequential(
            Block(64, 32, 64),
            Block(64, 32, 64),
            Block(64, 32, 64)
        )
        self.block2 = nn.Sequential(
            Block(64, 64, 128, True),
            Block(128, 64, 128),
            Block(128, 64, 128),
            Block(128, 64, 128)
        )
        self.block3 = nn.Sequential(
            Block(128, 128, 256, True),
            Block(256, 128, 256),
            Block(256, 128, 256),
            Block(256, 128, 256),
            Block(256, 128, 256),
            Block(256, 128, 256)
        )
        self.block4 = nn.Sequential(
            Block(256, 256, 512, True),
            Block(512, 256, 512),
            Block(512, 256, 512)
        )
        self.pool = nn.AvgPool2d(4, 4)
        self.fcn = nn.Sequential(
            nn.Linear(2 * 2 * 512, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.toplayer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fcn(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, planes, out_channels, dotted=False):
        super(Block, self).__init__()
        self.dotted = dotted
        if dotted:
            self.in_layer = nn.Sequential(
                BasicConv2d(in_channels, planes, 1),
                BasicConv2d(planes, planes, 3, stride=2, padding=1),
                BasicConv2d(planes, out_channels, 1)
            )
            self.dotted_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.in_layer = nn.Sequential(
                BasicConv2d(in_channels, planes, 1),
                BasicConv2d(planes, planes, 3, padding=1),
                BasicConv2d(planes, out_channels, 1)
            )

        self.out_layer = nn.Sequential(
            BasicConv2d(out_channels, planes, 1),
            BasicConv2d(planes, planes, 3, padding=1),
            nn.Conv2d(planes, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)


    def forward(self, x):
        residual = x
        if self.dotted:
            residual = self.dotted_layer(residual)
        out = self.in_layer(x)
        out = self.out_layer(out)
        out += residual
        out = self.relu(out)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ** kwargs):
        super(BasicConv2d, self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.step(x)



# net = ResNet()
# x = torch.randn(1, 3, 64, 64)
# y = net(Variable(x))
# print(y.size())
