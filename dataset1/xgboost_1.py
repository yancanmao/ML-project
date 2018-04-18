# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:38:29 2018

@author: lenovo
"""

import xgboost as xgb
import numpy as np
import sys
import csv
from xgboost import plot_tree
import matplotlib.pyplot as plt

# def loadCSVfile():
#       return np.loadtxt("data/heart_disease_all14.csv", dtype=np.str, delimiter=",")

# def loadData(filename):
#     data_reader = csv.reader(open(filename,'r'))
#     data = []
#     for row in data_reader:
#         data.append(row)
#     data = np.asarray(data,dtype='float32')
#     return data

# rawdata = loadCSVfile()

rawdata = np.load('new_heart_disease.npy')
# print(rawdata[:1])
# sys.exit(0)
max = 0
x=0
y=0
# for i in range(0,12):
# 	for j in range(i,12):
data = xgb.DMatrix(rawdata[:,:-1],label=(rawdata[:,-1]>0))
params = {
            'booster':'gbtree',
            'objective':'multi:softmax',
            'num_class':2,
            'eta':0.8,
            'max_depth':10,
            'subsample':1.0,
            'min_child_weight':5,
            'colsample_bytree':0.75,
            'scale_pos_weight':0.9,
            'eval_metric':'merror',
            'gamma':0.2,
            'lambda':300
}
num_round = 100
ret = xgb.cv(params,data,num_boost_round=num_round,nfold=10)
# if 1-ret.iloc[num_round-1][0] > max:
# 	max = 1-ret.iloc[num_round-1][0]
# 	x=i
# 	y=j
print('test-auc: ',1-ret.iloc[num_round-1][0])

# plot_tree(ret)

