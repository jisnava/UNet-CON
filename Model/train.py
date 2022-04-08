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
from loss import BCE

from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn import model_selection
import re

from sklearn.model_selection import KFold



def train_fn(model, train_loader,optimizer,DEVICE,scheduler):    
    model.train()
        
    average_loss = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):

        inputs = data["cm_others"]
        labels = data["contact_map"]
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE, dtype = torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = BCE(outputs[0][0], labels[0])
        loss.backward()
        average_loss += loss.item()
        optimizer.step()
    return average_loss/len(train_loader)