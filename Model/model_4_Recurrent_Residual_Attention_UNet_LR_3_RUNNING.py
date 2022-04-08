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
    def __init__(self,channel_input,channel_output):
        super(four_Convolutional_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_output, channel_output, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_output, channel_output, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_output, channel_output, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
    
class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

#-------------Attention UNet-----------------------------       
class Adding_Attention(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Adding_Attention,self).__init__()
        self.Weight_up = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.Weights_X = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.Weight_up(g)
        x1 = self.Weights_X(x)
        final_layer = self.relu(g1+x1)
        final_layer = self.final_layer(final_layer)

        return x*final_layer
    
class Upsample(nn.Module):
    def __init__(self,channel_input,channel_output):
        super(Upsample,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_input,channel_output,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(channel_output),
			nn.ReLU(inplace=True),
            nn.Conv2d(channel_output,channel_output,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(channel_output),
			nn.ReLU(inplace=True)
        )

    def forward(self,x1, x2):
        x1 = self.up(x1)
#         print(x.shape)
        Difference_Y = x2.size()[2] - x1.size()[2]
        Difference_Y = x2.size()[3] - x1.size()[3]
        #         x1 = F.pad(x1, [Difference_Y // 2, Difference_Y - Difference_Y // 2,
#                         Difference_Y // 2, Difference_Y - Difference_Y // 2])
        x1 = torch.nn.functional.pad(x1, [Difference_Y // 2, Difference_Y - Difference_Y // 2,
                        Difference_Y // 2, Difference_Y - Difference_Y // 2])
        return x1


class Attention_UNET(nn.Module):
    def __init__(self,img_ch=57,output_ch=1):
        super(Attention_UNET,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.recc1 = Recurrent_block(57, 2)
        self.Conv1 = four_Convolutional_block(channel_input=img_ch,channel_output=64)
        self.recc2 = Recurrent_block(64, 2)
        self.Conv2 = four_Convolutional_block(channel_input=64,channel_output=128)
        self.recc3 = Recurrent_block(128, 2)
        self.Conv3 = four_Convolutional_block(channel_input=128,channel_output=256)
        self.recc4 = Recurrent_block(256, 2)
        self.Conv4 = four_Convolutional_block(channel_input=256,channel_output=512)
        self.recc5 = Recurrent_block(512, 2)
        self.Conv5 = four_Convolutional_block(channel_input=512,channel_output=1024)

        self.Up5 = Upsample(channel_input=1024,channel_output=512)
        self.Att5 = Adding_Attention(F_g=512,F_l=512,F_int=256)
        self.Upsample5 = four_Convolutional_block(channel_input=1024, channel_output=512)

        self.Up4 = Upsample(channel_input=512,channel_output=256)
        self.Att4 = Adding_Attention(F_g=256,F_l=256,F_int=128)
        self.Upsample4 = four_Convolutional_block(channel_input=512, channel_output=256)
        
        self.Up3 = Upsample(channel_input=256,channel_output=128)
        self.Att3 = Adding_Attention(F_g=128,F_l=128,F_int=64)
        self.Upsample3 = four_Convolutional_block(channel_input=256, channel_output=128)
        
        self.Up2 = Upsample(channel_input=128,channel_output=64)
        self.Att2 = Adding_Attention(F_g=64,F_l=64,F_int=32)
        self.Upsample2 = four_Convolutional_block(channel_input=128, channel_output=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        # encoding path
        x1 = self.recc1(x)
        x1 = self.Conv1(x)


        x2 = self.Maxpool(x1)
        x2 = self.recc2(x2)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.recc3(x3)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.recc4(x4)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.recc5(x5)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5, x4)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Upsample5(d5)
#         print(d5.shape)
        
        d4 = self.Up4(d5, x3)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Upsample4(d4)

        d3 = self.Up3(d4, x2)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Upsample3(d3)

        d2 = self.Up2(d3, x1)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Upsample2(d2)

        d1 = self.Conv_1x1(d2)

        return self.sigmoid(d1)
    

