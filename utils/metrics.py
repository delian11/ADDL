import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import sklearn.metrics as sm


def Get_confusion_matrix(true_label, pre_label):
    acc = np.count_nonzero( np.equal(pre_label,true_label) ) * 1. / true_label.shape[0]
    acc_mat = []
    for i in np.unique(true_label):
        index = np.where(true_label==i)
        class_pre = pre_label[index]
        class_num = len(index[0])
        class_mat = []
        for j in np.unique(true_label):
            pre_class_num = len(np.where(class_pre==j)[0]) *1.0
            class_mat.append(round(pre_class_num / class_num, 3))
        acc_mat.append(class_mat)
    acc_mat=np.array(acc_mat)
    return acc_mat,acc

