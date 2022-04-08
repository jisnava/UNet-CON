#INFERENCE
#from model_2_UNet_Residual_LR_3_RUNNING import UNet #UNet #,,
from model_3_Attention_UNet_LR_2_RUNNING import Attention_UNET #UNet #,,
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from torch.optim.optimizer import Optimizer
import random
import time
from dataset import Dataset_class
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from dataset import total_features, Dataset_class
from metrics import topKaccuracy ,evaluate, output_result 
from train import train_fn
from eval import eval_fn

from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn import model_selection
import re
test_path="pdb25-test-500.release.contactFeatures.pkl"
# from args import config
# arg=config
# train_path=arg.train_path
# valid_path=arg.valid_path
# test_path=arg.test_path
# lr= arg.lr 
# k_splits =  arg.k_splits 
# max_len = arg.max_len 
# pad_size = arg.pad_size 
# num_blocks =  arg.num_blocks 
# expected_n_channels = arg.expected_n_channels 
# protein_len_consider = arg.protein_len_consider 
# weight_decay_1 = arg.weight_decay_1 
# weight_decay_2 = arg.weight_decay_2
# epochs= arg.epochs
# best_val_min = arg.best_val_min 


from sklearn.model_selection import KFold

with open(test_path, 'rb') as f:
    protein_test_file = pickle.load(f, encoding = "latin1")
for i, data in tqdm(enumerate(protein_test_file), total = len(protein_test_file)):
    data["contactMatrix"] = np.where(data["contactMatrix"] <=0 , 0, 1)

test_list = []
for line in protein_test_file:
    #print(line["contactMatrix"])
    train_dict = {}

    train_dict ={
        "sequence" : line["sequence"],
        "ACC" : line["ACC"],
        "PSFM" : line["PSFM"],
        "ccmpredZ" : line["ccmpredZ"],
        "PSSM" : line["PSSM"],
        "SS3" : line["SS3"],
        "SS8" : line["SS8"],
        "psicovZ" : line["psicovZ"],
        "OtherPairs" : line["OtherPairs"],
        "DISO" : line["DISO"],
        "contact_map" : line["contactMatrix"]
    }
    if(line["contactMatrix"].shape[0] > 50):
        test_list.append(train_dict)

#model_1 = UNet(57, 1)
model_1 = Attention_UNET(57, 1)
model_1.load_state_dict(torch.load("/workstation/jisna/saved_weights/model_3_Attention_UNet_LR_3.pth", map_location = 'cuda:0'))
model_1.cuda()
test_dataset = Dataset_class(test_list,128,57,0)
test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 4)
DEVICE = torch.device("cuda")
import torch.nn.functional as F
f_1score, precesion_score, recall_score, l_1, l_2, l_5, l_10, score_list_1 = eval_fn(model_1,test_loader, DEVICE)
print(f_1score, precesion_score, recall_score, output_result(score_list_1))