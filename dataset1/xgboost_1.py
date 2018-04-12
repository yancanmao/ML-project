# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:38:29 2018

@author: lenovo
"""

import xgboost as xgb
import numpy as np
import sys


rawdata = np.load('new_heart_disease.npy')
# print(rawdata[:1, 0:1])
# sys.exit(0)
max = 0
x=0
y=0
# for i in range(0,12):
# 	for j in range(i,12):
data = xgb.DMatrix(rawdata[323:626,:-1],label=(rawdata[323:626,-1]>0)) # cleveland only
params = {
            'booster':'gbtree',
            'objective':'multi:softmax',
            'num_class':5,
            'eta':0.1,
            'max_depth':10,
            'subsample':1.0,
            'min_child_weight':5,
            'colsample_bytree':0.2,
            'scale_pos_weight':0.1,
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
