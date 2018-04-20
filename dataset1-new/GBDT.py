# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:42:29 2018

@author: lenovo
"""

import lightgbm as lgb
import numpy as np
#from scipy.io.arff import loadarff
import sys
from matplotlib.pyplot import savefig
import csv
#import os

filename = 'heart_disease'


def loadData(filename):
    data = np.load(filename+'.npy')
    types = ['AGE','SEX','CP','THRESTBPS','CHOL','FBS','RESTECG','THALACH',
             'EXANG','OLDPEAK','SLOPE','CA','THAL','CATEGORY']
    return data, types

def train():
    data, types = loadData(filename)
    dataset = lgb.Dataset(data[:,:-1],label=(data[:,-1]>0),feature_name=types[:-1])
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'metric_freq': 100,
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.02,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    
    model_cv = lgb.cv(params,dataset,num_boost_round=120,nfold=10)
    print('10-fold cross validation accuracy: %.3f: '%model_cv['auc-mean'][-1])
    
    model = lgb.train(params, dataset, num_boost_round=120)
    
    print('printing tree structure...')
    try:
        #graph = lgb.create_tree_digraph(model)
        #graph.render('tree.gv',view=True)
        #os.system('dot tree.gv -Tpng -o tree.png')
        lgb.plot_tree(model,figsize=(100,25))
        savefig('tree.png')
    except:
        print('You must install graphviz to plot tree.\
              \nPlease visit http://www.graphviz.org/download/ \
              \nor "conda install graphviz" to download the software\
              \nthen "pip install graphviz" to install python binding')
    finally:
        pass
    
    lgb.plot_importance(model,height=0.5,figsize=(15,10))
    print('printing importance histogram...')
    savefig('importance.png')
    
    model.save_model('gbdtmodel.txt')
    print('model saved in "gbdtmodel.txt".')


def gbdt_classify(data_file,model_file='gbdtmodel.txt'):
    model = lgb.Booster(model_file=model_file)
    new_data = loadData(data_file)
    data_len = int(np.size(new_data,0))
    ret = model.predict(new_data).reshape(data_len)
    newname = data_file.split('.')
    new_name = newname[0]+'_output.'+newname[-1]
    out = open(new_name, 'w', newline='')
    csv_writer = csv.writer(out, dialect='excel')
    for i in range(data_len):
        csv_writer.writerow([ret[i]])
    print('Results written in '+new_name+'.')


if __name__=='__main__':
    if len(sys.argv) == 1:
        train()
    if len(sys.argv) == 2:
        gbdt_classify(sys.argv[1])
    if len(sys.argv) == 3:
        gbdt_classify(sys.argv[1],sys.argv[2])
    