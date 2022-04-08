from model_1_UNet_LR_3_RUNNING_FINE import UNet #Attention_UNET #,,
#Done-model_1_UNet_LR_3, model_2_UNet_Residual_LR_3, model_3_Attention_UNet_LR_2, model_4_Recurrent_Residual_Attention_UNet_LR_3, model_5_Residual_Attention_UNet_LR_3
from dataset import total_features, Dataset_class
from metrics import topKaccuracy ,evaluate, output_result 
from train import train_fn
from eval import eval_fn
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import KFold
from transformers import get_cosine_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
from pandas import DataFrame
from loss import BCE
from args import config
arg=config
train_path=arg.train_path
valid_path=arg.valid_path
test_path=arg.test_path
lr= arg.lr 
k_splits =  arg.k_splits 
max_len = arg.max_len 
pad_size = arg.pad_size 
num_blocks =  arg.num_blocks 
expected_n_channels = arg.expected_n_channels 
protein_len_consider = arg.protein_len_consider 
weight_decay_1 = arg.weight_decay_1 
weight_decay_2 = arg.weight_decay_2
epochs= arg.epochs
best_val_min = arg.best_val_min 


with open(train_path, 'rb') as f:
    protein_train_file = pickle.load(f, encoding = "latin1")

with open(valid_path, 'rb') as f:
    protein_valid_file = pickle.load(f, encoding = "latin1")

for i, data in tqdm(enumerate(protein_train_file), total = len(protein_train_file)):
    data["contactMatrix"] = np.where(data["contactMatrix"] <=0 , 0, 1)

train_list = []
for line in protein_train_file:
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
    if(line["contactMatrix"].shape[0] > protein_len_consider):
        train_list.append(train_dict)

len(train_list)

for i, data in tqdm(enumerate(protein_valid_file), total = len(protein_valid_file)):
    data["contactMatrix"] = np.where(data["contactMatrix"] <=0 , 0, 1)

for line in protein_valid_file:
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
    if(line["contactMatrix"].shape[0] > protein_len_consider):
        train_list.append(train_dict)

train_array = np.asarray(train_list)
kf = KFold(n_splits=k_splits, random_state=42, shuffle=True)
all_folds = {}
for fold, (train_index, test_index) in enumerate(kf.split(train_array)):
    train, test = train_array[train_index], train_array[test_index]
    all_folds[fold] = [train, test]



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

len(test_list)
loss_1=[]
epoch_1=[]

def run(all_fold, test_list,fold):
    fold=1

    
    train_dataset = Dataset_class(all_fold[fold][0], max_len, expected_n_channels, pad_size)
    test_dataset = Dataset_class(all_fold[fold][1], max_len, expected_n_channels, pad_size)

    train_loader = DataLoader(train_dataset, batch_size = 1, num_workers = 4)
    valid_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 4)
    
    test=Dataset_class(test_list, max_len, expected_n_channels, pad_size)
    test_loader=DataLoader(test, batch_size = 1, num_workers = 4)

    DEVICE = torch.device("cuda:0")
    
#     model= Attention_UNET(57,1)
    model= UNet(57,1)

    model.to(DEVICE)


    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_1},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_2},
    ]
    num_train_steps = int(len(train_list) / 4 * epochs)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_train_steps
        )

    best_val = best_val_min 
    for epoch in range(epochs):
        loss = train_fn(model,train_loader, optimizer, DEVICE,scheduler)
        loss_1.append(loss)
        epoch_1.append(epoch)
        print("epochs", epoch)
        print("=========================================")
        print("average_training_loss ", loss)

        f_1score, precesion_score, recall_score, l_1, l_2, l_5, l_10, score_lis = eval_fn(model,valid_loader, DEVICE)
        f_1score_1, precesion_score_1, recall_score_1, l_1_1, l_2_1, l_5_1, l_10_1, score_lis_1 = eval_fn(model,test_loader, DEVICE)
     
    
#         print("=============f_1 loss========", f_1score)
#         print("=============precesion_score========", precesion_score)
#         print("=============recall_score========", recall_score)
        
#         print("===========Train_Valid============")  
#         print(output_result(score_lis))
        # print("===========Test_500============")
        # print(output_result(score_lis_1))
        if score_lis_1[3][0] >= best_val:
            
            print("===================Saving model================")
            best_val = score_lis_1[3][0]
            model_path = f"/workstation/jisna/saved_weights/model_1_UNet_LR_3.pth"
            torch.save(model.state_dict(), model_path)
# -
if __name__ == "__main__":
    run(all_folds, test_list,1)
    submission=pd.DataFrame(list(zip(epoch_1,loss_1)),columns=["epoch","loss"])
    submission.to_csv("model_1_UNet_LR_3_variable_length_all_57_features_March_15.csv",index=False)
# -


