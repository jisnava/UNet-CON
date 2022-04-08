import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from torch.optim.optimizer import Optimizer
import random
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm

from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn import model_selection
import re

from sklearn.model_selection import KFold

class four_Convolutional_block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

#-------------UNet-Residual-----------------------------    
class MaxPooling(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(2)
        self.dconv1=four_Convolutional_block(in_channels, out_channels)
        self.dconv2=four_Convolutional_block(out_channels, out_channels)
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             four_Convolutional_block(in_channels, out_channels),
#             four_Convolutional_block(out_channels, out_channels)
#         )


    def forward(self, x):
        x=self.maxpool_conv(x)
        x=self.dconv1(x)
        x1=x
        x=self.dconv2(x)
        x=x1+x
        return x


class Upsample(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = four_Convolutional_block(in_channels, out_channels, in_channels // 2)
            self.conv2 = four_Convolutional_block(out_channels, out_channels, out_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv1 = four_Convolutional_block(in_channels, out_channels)
            self.conv2 = four_Convolutional_block(out_channels, out_channels)

    def forward(self,x1, x2):
        x1 = self.up(x1)
#         print(x.shape)
        Difference_Y = x2.size()[2] - x1.size()[2]
        Difference_Y = x2.size()[3] - x1.size()[3]
        #         x1 = F.pad(x1, [Difference_Y // 2, Difference_Y - Difference_Y // 2,
#                         Difference_Y // 2, Difference_Y - Difference_Y // 2])
        x1 = torch.nn.functional.pad(x1, [Difference_Y // 2, Difference_Y - Difference_Y // 2,
                        Difference_Y // 2, Difference_Y - Difference_Y // 2])
        x = torch.cat([x2, x1], dim=1)
        x= self.conv1(x)
        x11=x
        x=self.conv2(x)
        x=x11+x
        return x


class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = four_Convolutional_block(n_channels, 64)
        self.down1 = MaxPooling(64, 128)
        self.down2 = MaxPooling(128, 256)
        self.down3 = MaxPooling(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = MaxPooling(512, 1024 // factor)
        self.up1 = Upsample(1024, 512 // factor, bilinear)
        self.up2 = Upsample(512, 256 // factor, bilinear)
        self.up3 = Upsample(256, 128 // factor, bilinear)
        self.up4 = Upsample(128, 64, bilinear)
        self.outc = OutConvolution(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.sigmoid(logits)
        return logits    






    
