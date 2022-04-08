import os
import numpy as np

# +
import os
import numpy as np

def topKaccuracy(y_out, y, k):
    L = y.shape[0]

    m = np.ones_like(y, dtype=np.int8)
    lm = np.triu(m, 24)
    mm = np.triu(m, 12)
    sm = np.triu(m, 6)
    
    #print("lm:",lm)
    #print("mm:",mm)
    #print("sm:",sm)
    #print("lm:",lm.shape)
    #print("mm:",mm.shape)
    #print("sm:",sm.shape)

    sm = sm - mm
    mm = mm - lm

    avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((avg_pred[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    #print("y_out:",y_out)
    #print("y_out:",y_out.shape)

    #print("y:",y)
    #print("y:",y.shape)

    #print("avg_pred:",avg_pred)
    #print("avg_pred:",avg_pred.shape)

    #print("truth:",truth)
    #print("truth:",truth.shape)

    accs = []
    recalls = []
    for x in [lm, mm, sm]:
        selected_truth = truth[x.nonzero()]
        selected_false = truth[np.where( x == 0 )]
        #print("selected_truth:",selected_truth)
        #print("selected_truth:",selected_truth.shape)
        selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort()[::-1]]
        selected_false_sorted = selected_false[(selected_false[:, 0]).argsort()[::-1]]
        #print("selected_truth_sorted:",selected_truth_sorted)
        #print("selected_truth_sorted:",selected_truth_sorted.shape)

        #print("selected_truth_sorted[:, 1]:",selected_truth_sorted[:, 1])
        #print("selected_truth_sorted[:, 1]:",selected_truth_sorted[:, 1].shape)

        tops_num = min(selected_truth_sorted.shape[0], L//k)
        tops_num1 = min(selected_false_sorted.shape[0], L//k)
        #print("tops_num:",tops_num)

        truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
        false_in_pred = selected_false_sorted[:, 1].astype(np.int8)
        #print("truth_in_pred:",truth_in_pred)
        #print("truth_in_pred:",truth_in_pred.shape)

        corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
        corrects_num1 = np.bincount(false_in_pred[0: tops_num1], minlength=2)
        #print("corrects_num:",corrects_num)
        #print("corrects_num:",corrects_num.shape)
        #print("corrects_num1:",corrects_num1)
        #print("corrects_num1:",corrects_num1.shape)

        acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
        recall = 1.0 * corrects_num[1] / (1.0 * corrects_num[1] + 1.0 * corrects_num1[1] + 0.0001)
        #print("acc:",acc)
        #print("acc:",acc.shape)
        #print("recall:",recall)
        #print("recall:",recall.shape)

        accs.append(acc)
        recalls.append(recall)
    #print("accs:",accs)
    #print("recalls:",recalls)

    return accs, recalls

def evaluate(predict_matrix, contact_matrix):
    acc_k_1, recall_k_1 = topKaccuracy(predict_matrix, contact_matrix, 1)
    acc_k_2, recall_k_2 = topKaccuracy(predict_matrix, contact_matrix, 2)
    acc_k_5, recall_k_5 = topKaccuracy(predict_matrix, contact_matrix, 5)
    acc_k_10, recall_k_10 = topKaccuracy(predict_matrix, contact_matrix, 10)
    tmp = []
    tmp1 =[]
    tmp.append(acc_k_1)
    tmp.append(acc_k_2)
    tmp.append(acc_k_5)
    tmp.append(acc_k_10)

    tmp1.append(recall_k_1)
    tmp1.append(recall_k_2)
    tmp1.append(recall_k_5)
    tmp1.append(recall_k_10)

    #print("tmp:",tmp)
    #print("tmp1:",tmp1)

    return tmp,tmp1

def output_result(avg_acc):
    print ("Long Range(> 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f"             %(avg_acc[3][0], avg_acc[2][0], avg_acc[1][0], avg_acc[0][0]))
    print ("Medium Range(12 - 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f"             %(avg_acc[3][1], avg_acc[2][1], avg_acc[1][1], avg_acc[0][1]))
    print ("Short Range(6 - 12):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f"             %(avg_acc[3][2], avg_acc[2][2], avg_acc[1][2], avg_acc[0][2]))

def output_result1(avg_acc):
    print ("Long Range(> 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Recall :     %.3f        %.3f        %.3f      %.3f"             %(avg_acc[3][0], avg_acc[2][0], avg_acc[1][0], avg_acc[0][0]))
    print ("Medium Range(12 - 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Recall :     %.3f        %.3f        %.3f      %.3f"             %(avg_acc[3][1], avg_acc[2][1], avg_acc[1][1], avg_acc[0][1]))
    print ("Short Range(6 - 12):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Recall :     %.3f        %.3f        %.3f      %.3f"             %(avg_acc[3][2], avg_acc[2][2], avg_acc[1][2], avg_acc[0][2]))
import numpy as np

def FCC(TP, FP, TN, FN):
    epsilon = 0.000001
    precision = TP*1./(TP+FP + epsilon)
    recall = TP*1./(TP+FN + epsilon)
    F1 = 2.*precision*recall/(precision + recall + epsilon)

    return F1, precision, recall


def MC(TP, FP, TN, FN):

    epsilon = 0.000001
    MCC = (TP*TN - FP*FN)/np.sqrt( epsilon + 1.0*(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )
    return MC


def CalcMCCF1(pred=None, truth=None, probCutoff=0.0, contactCutoff=8.0):
    if pred is None:
        print ('please provide a predicted contact matrix')
        exit(-1)

    if truth is None:
        print ('please provide a true distance matrix')
        exit(-1)

    assert pred.shape == truth.shape

    ## in case the matrix is not square, e.g., interfacial contact matrix
    seqLen = pred.shape[0]
    seqLen2 = pred.shape[1]

    pred_binary = (pred>probCutoff)
    truth_binary = ( 0<truth) & (truth<contactCutoff )
    pred_truth = pred_binary * 2 + truth_binary
    numPredicted = np.sum(pred_binary)
    numTruths = np.sum(truth_binary)
    #print "#predicted=", numPredicted, "#natives=", numTruths

    mask_LR = np.triu_indices(seqLen, 24, m=seqLen2)
    mask_MLR = np.triu_indices(seqLen, 12, m=seqLen2)
    mask_SMLR = np.triu_indices(seqLen, 6, m=seqLen2)


    metrics = []
    for mask in [ mask_LR, mask_MLR, mask_SMLR]:

        res = pred_truth[mask]
    total = res.shape[0]
    count = np.bincount(res, minlength=4)
    assert (total == np.sum(count) )

    ## pred=0, truth=0	
    TN = count[0]

    ## pred=0, truth=1
    FN = count[1]

    ## pred=1, truth=0
    FP = count[2]

    ## pred=1, truth=1
    TP = count[3]

    #print TP, FP, TN, FN

    MCC = MC(TP, FP, TN, FN)
    F1, precision, recall = FCC(TP, FP, TN, FN)

    metrics.extend ([MCC, TP, FP, TN, FN, F1, precision, recall])


    return metrics