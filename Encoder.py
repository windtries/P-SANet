#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import init
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.1)
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        y = avg_out + max_out
        y = self.relu2(y)
        y = x*y
        return y

class MASM(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _= torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y1 = self.conv1(y)
        y2 = self.conv2(y)
        y3 = self.conv3(y)
        y = y1+y2+y3
        y = self.relu(y)
        y = x*y
        return y

# BasicBlock continues several PCEM sub-modules
class BasicBlock(nn.Module):
  def __init__(self, kernel_size, in_sizes, out_sizes, stride, ca, sa):
    super(BasicBlock, self).__init__()
    self.stride = stride
    self.out_size =out_sizes[1]
    init_channels = math.ceil(out_sizes[1] / 4)
    new_channels = init_channels*3
    self.ca1 = ca
    self.ca2 = ChannelAttention(new_channels)
    self.sa = sa
    self.conv1 = nn.Conv2d(in_sizes, out_sizes[0], kernel_size=1, stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(out_sizes[0])
    self.nolinear1 = nn.LeakyReLU(0.1)
    self.primary_conv = nn.Sequential(
        nn.Conv2d(out_sizes[0], init_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(init_channels),
        nn.ReLU(inplace=True),
    )

    self.cheap_operation = nn.Sequential(
        nn.Conv2d(init_channels, new_channels,kernel_size, 1, kernel_size // 2, groups=init_channels, bias=False),
        nn.BatchNorm2d(new_channels),
        nn.ReLU(inplace=True),
    )
    self.bn2 = nn.BatchNorm2d(out_sizes[1])
    self.nolinear2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x
    x = self.nolinear1(self.bn1(self.conv1(x)))
    x1 = self.primary_conv(x)
    x2 = self.cheap_operation(x1)
    x2 = self.ca2(x2)
    x = torch.cat([x1, x2], dim=1)
    x = x[:, :self.out_size, :, :]
    x = self.nolinear2(self.bn2(x))
    x += residual
    if self.sa!=None:
        x = self.sa(x)
    if self.ca1!=None:
        x = self.ca1(x)
    return x

# number of layers per model
model_blocks = {
    1: [1, 3, 3, 4, 1],
    2: [2, 3, 3, 3, 2],
    3: [2, 4, 5, 6, 2],
    4: [3, 5, 8, 8, 6]
}


class Backbone(nn.Module):


  def __init__(self, params):
    super(Backbone, self).__init__()
    self.use_range = params["input_depth"]["range"]
    self.use_xyz = params["input_depth"]["xyz"]
    self.use_remission = params["input_depth"]["remission"]
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]
    self.OS = params["OS"]
    self.layers = params["extra"]["layers"]

    # input depth calc
    self.input_depth = 0
    self.input_idxs = []
    if self.use_range:
      self.input_depth += 1
      self.input_idxs.append(0)
    if self.use_xyz:
      self.input_depth += 3
      self.input_idxs.extend([1, 2, 3])
    if self.use_remission:
      self.input_depth += 1
      self.input_idxs.append(4)
    print("Depth of backbone input = ", self.input_depth)
    self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.LeakyReLU(0.1)

    #     # stride play
    self.strides = [2, 2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    # make the new stride
    if self.OS > current_os:
      print("Can't do OS, ", self.OS,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == self.OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)

    assert self.layers in model_blocks.keys()

    self.blocks = model_blocks[self.layers]

    # encoder
    self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                     stride=self.strides[0], kernel_size=3, ca=None, sa=MASM())
    self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                     stride=self.strides[1], kernel_size=3, ca=None, sa=MASM())
    self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                     stride=self.strides[2], kernel_size=3, ca=None, sa=None)
    self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                     stride=self.strides[2], kernel_size=3, ca=None, sa=None)
    self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                     stride=self.strides[2], kernel_size=3, ca=ChannelAttention(1024), sa=None)

    self.dropout = nn.Dropout2d(self.drop_prob)
    # last channels
    self.last_channels = 1024

  # make layer useful function
  def _make_enc_layer(self, block, planes, blocks, stride, kernel_size, ca, sa):
    layers = []
    #  downsample
    layers.append(("conv2", nn.Conv2d(planes[0], planes[1],
                                      kernel_size=kernel_size,
                                      stride=[1, stride],
                                      padding=kernel_size // 2, bias=False)))
    layers.append(("bn2", nn.BatchNorm2d(planes[1])))
    layers.append(("nl2", nn.LeakyReLU(0.1)))
    #  blocks
    inplanes = planes[1]
    for i in range(0, blocks):
      layers.append(("residual_{}".format(i),
                     block(kernel_size, inplanes, planes, stride=1, ca=ca, sa=sa)))
    return nn.Sequential(OrderedDict(layers))

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2      
    x = y
    return x, skips, os

  def forward(self, x):
    # filter input
    x = x[:, self.input_idxs]
    # run cnn
    # store for skip connections
    skips = {}
    os = 1
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.relu1, skips, os)
    x, skips, os = self.run_layer(x, self.enc1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc4, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc5, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth