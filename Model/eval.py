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
from metrics import *

from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn import model_selection
import re

from sklearn.model_selection import KFold


def eval_fn(model, valid_loader, DEVICE):
    model.eval()
    average_loss = 0
    precision = []
    recall = []
    f_1 = []
    acc = []
    L_1 = []
    L_2 = []
    L_5 = []
    L_10 = []
    score_list = np.zeros((4, 3))
    for i, data in tqdm(enumerate(valid_loader), total = len(valid_loader)):

        cm_others = data["cm_others"]
        target = data["contact_map"]
        cm_others = cm_others.to(DEVICE)
        target = target.to(DEVICE, dtype = torch.float)
    #     print(cm_others.shape)

    #     print(cm_others[0][0])
    #     optimizer.zero_grad()
        outputs = model(cm_others)
    #     print(outputs.shape)
    #     print(target.shape)

        target = target.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

#             scores = evaluate(outputs[0], target[0])
#             acc.append(np.mean(scores[0]))
#             recall.append(np.mean(scores[1]))
#         print(outputs.shape)
#         print(target.shape)
        f_1.append(CalcMCCF1(outputs[0][0], target[0],probCutoff = 0.52 )[5])
        precision.append(CalcMCCF1(outputs[0][0], target[0], probCutoff = 0.52)[6])
        recall.append(CalcMCCF1(outputs[0][0], target[0], probCutoff = 0.52)[7])
        sample_calculate = CalcMCCF1(outputs[0][0], target[0], probCutoff = 0.52)
        score = evaluate(outputs[0][0], target[0])
#             L_1.append(score[0][0])
#             L_2.append(score[0][1])
#             L_5.append(score[0][2])
#             L_10.append(score[0][3])
        score_list = np.add(score_list, np.asarray(score[0]))
    return np.mean(f_1), np.mean(precision), np.mean(recall), L_1, L_2, L_5, L_10, score_list/len(valid_loader)
