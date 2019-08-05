# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 23:07:53 2019

@author: Regory Gregory
not really efficient if you want to use all the metrics as it recalculates tp,fp,tn,fn...etc.
#forget about self
#how do static variables work?

"""
import numpy as np
import tensorflow.keras.backend as KB

threshold = 0.5


def calc_precision(y_true, y_pred):
    y_pred = KB.round(y_pred)
    #detections = tp+fp
    detections = KB.sum(y_pred)
    tp = KB.sum(KB.clip(y_pred*y_true, 0, 1))
    
    return tp/detections


def calc_recall(y_true, y_pred):
    #fp/fp+fn
    y_pred = KB.round(y_pred)
    tp = KB.sum(KB.clip(y_pred*y_true, 0, 1))
    
    y_pred_inv = KB.abs(y_pred-1)
    y_true_inv = KB.abs(y_true-1)
    
    tn = KB.sum(y_pred_inv*y_true_inv)
    n_all = KB.sum(y_pred_inv)
    
    fn = n_all-tn
    
    return tp/(tp+fn)   

def calc_f1(y_true, y_pred):

    precision = calc_precision(y_true, y_pred)
    recall = calc_recall(y_true, y_pred)

    return 2*(precision*recall)/(precision+recall)
    