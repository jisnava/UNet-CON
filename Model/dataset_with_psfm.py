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


def total_features(features, expected_channels):
    l = len(features["sequence"])
    features_list = np.full((l, l, expected_channels), 0.0)
    acc = features["ACC"]
    fi = 0
    for j in range(3):
        a = np.repeat(acc[:,j].reshape(1, l), l, axis = 0)
        features_list[:, :, fi] = a
        fi += 1
        features_list[:, :, fi] = a.T
        fi += 1
    pssm = features["PSSM"]
    
    ss3 = features["SS3"]
    for j in range(3):
        a = np.repeat(ss3[:,j].reshape(1, l), l, axis = 0)
        features_list[:, :, fi] = a
        fi += 1
        features_list[:, :, fi] = a.T
        fi += 1

    for j in range(20):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis = 0)
        features_list[:, :, fi] = a
        fi += 1
        features_list[:, :, fi] = a.T
        fi += 1
        
    psfm = features["PSFM"]
    for j in range(20):
        a = np.repeat(psfm[:, j].reshape(1, l), l, axis = 0)
        features_list[:, :, fi] = a
        fi += 1
        features_list[:, :, fi] = a.T
        fi += 1

    ccmpred = features["ccmpredZ"]
    features_list[:, :, fi] = ccmpred
    fi += 1

#     psicovZ = features["psicovZ"]
#     features_list[:, :, fi] = psicovZ
#     fi +=1

    OtherPairs = features["OtherPairs"]
    features_list[:, :, fi:fi+3] = OtherPairs
    fi += 3
    
    return features_list

# +
class Dataset_class(Dataset):
    def __init__(self, train_list, max_len, expected_channels, pad_size):
        super().__init__()
        self.train_list = train_list
        self.max_len = max_len
        self.expected_channels = expected_channels
        self.pad_size = pad_size

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):  
        
        features = self.train_list[index]
        
        XX = np.full( (self.max_len, self.max_len, self.expected_channels), 0.0)
#         print(XX.shape)
        YY = np.full((self.max_len, self.max_len, 1), 0)

        features = self.train_list[index]

        # cm_others has total of 6 features from dataset which includes
        # PSSM, SS3 , ACC, CCMPRED, PSICOV, OTHERPAIR
        # Converting this 6 features in the form of (L, L, 57) 
        # as shown in the total_features function
        
        cm_others = total_features(features,self.expected_channels )
        
        target = features["contact_map"]
        cm_others = np.transpose(cm_others, (2, 0, 1))

        return {
            "cm_others" : torch.tensor(cm_others, dtype = torch.float), 
            "contact_map" : torch.tensor(target, dtype = torch.float)
        }