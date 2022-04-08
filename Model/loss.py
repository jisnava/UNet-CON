import torch
import torch.nn as nn
def BCE(outputs,target, alpha = 1, gamma = 2):
    loss = nn.BCELoss()
    loss = loss(outputs, target)
    return loss