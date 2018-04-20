# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:00:09 2018

@author: lenovo
"""
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt
#from scipy.signal import medfilt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Masking
from keras.optimizers import Adam
#import lightgbm as lgb


fft_length = 120
pulse_length = 400


def list_files(directory):
    
    return [item for item in os.listdir(directory)
            if item.startswith('normal') or \
               item.startswith('extrastole') or \
               item.startswith('murmur') or \
               item.startswith('extrahls')]


def get_pulse_feature(x,pulse_length=pulse_length):
    fs = x[0]
    x_abs = np.abs(x[1])
    # a heart beat pulse is shorter than 0.075s. 
    pulse_feature = -np.ones(pulse_length)
    valid_pos = np.where(x_abs>-1)[0]
    valid_pos = (np.floor(valid_pos/(fs*0.075))).astype(int)
    pulse_feature[valid_pos] = 0
    pulse_pos = np.where(x_abs>np.minimum(np.max(x_abs)*0.4,np.median(x_abs)*15))[0]
    pulse_pos = (np.floor(pulse_pos/(fs*0.075))).astype(int)
    pulse_feature[pulse_pos] = 1
    
    return pulse_feature

    
def load_data(location):
    contents = list_files(location)
    dataframe = pd.DataFrame(contents, columns=['file_name'])
    
    # assign label to dataframe
    dataframe['normal'] = dataframe['file_name'].apply(lambda x: x.startswith('normal')).astype(int)
    dataframe['fft_normal'] = dataframe['file_name'].apply(lambda x: not x.startswith('murmur')).astype(int)
    dataframe['pulse_normal'] = dataframe['file_name'].apply(lambda x: not x.startswith('extra')).astype(int)
      
    if not location.endswith('/'):
        location += '/'
    
    dataframe['file_name'] = location + dataframe['file_name']
    dataframe['set_name'] = dataframe['file_name'].apply(lambda x: "b" if "set_b" in x else "a")
    dataframe['raw_read'] = dataframe['file_name'].apply(wav.read)
    dataframe['sample_frequency'] = dataframe['raw_read'].apply(lambda x: x[0])
    dataframe['wav_file_data'] = dataframe['raw_read'].apply(lambda x: x[1])
    dataframe['data_length'] = dataframe['raw_read'].apply(lambda x: len(x[1]))
    #dataframe['fft_feature'] = dataframe['wav_file_data'].apply(lambda x: medfilt(np.abs(fft(x,2048))[:fft_length]/np.max(np.abs(fft(x,2048))),15))
    dataframe['fft_feature'] = dataframe['raw_read'].apply(lambda x: np.abs(fft(x[1],2*int(1024*4000/x[0])))[:fft_length]/np.max(np.abs(fft(x[1],2048))))
    dataframe['pulse_feature'] = dataframe['raw_read'].apply(get_pulse_feature)
    dataframe['seconds'] = dataframe['data_length'] / dataframe['sample_frequency']
    
    return dataframe


def shuffle_dataframe(dataframe):
    """
    Shuffles the contents of a pandas dataframe and returns the
    shuffled dataframe
    """
    
    return dataframe.sample(frac=1).reset_index(drop=True)


def get_label_rates(dataframe):
    normal_rate = (dataframe['normal'] == 1).astype(int).sum()# / len(dataframe)
    murmur_rate = (dataframe['fft_normal'] == 0).astype(int).sum()# / len(dataframe)
    extra_rate = (dataframe['pulse_normal'] == 0).astype(int).sum()# / len(dataframe)
    
    return normal_rate, murmur_rate, extra_rate


def fft_model(fft_length=fft_length):
    model = Sequential()
    model.add(LSTM(8, input_shape=(1, fft_length), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(4, dropout=0.2, recurrent_dropout=0.2))
#    model.add(Dense(24, input_shape=(fft_length,)))
#    model.add(Activation('selu'))
#    model.add(Dense(6))
#    model.add(Activation('selu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    
    return model


def pulse_model(pulse_length=pulse_length):
    model = Sequential()
    model.add(Masking(mask_value= -1,input_shape=(1, pulse_length)))
    model.add(LSTM(8, input_shape=(1, pulse_length), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(8, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    
    return model
    

# Set A is from random people using iStethoscope Pro iPhone app
# 44100Hz
dataframe_a = load_data('./heartbeat-sounds/set_a')
#print(get_label_rates(dataframe_a))

# Set B is from clinical trial
# 4000Hz
dataframe_b = load_data('./heartbeat-sounds/set_b')
#print(get_label_rates(dataframe_b))

# concat is a good idea, maybe I should do it.
b_len = len(dataframe_b)
a_len = len(dataframe_a)
examples = a_len + b_len

#First model
X = -np.ones([examples, fft_length])
Y = np.zeros([examples,1])
Y[:a_len,0] = dataframe_a['fft_normal']
Y[-b_len:,0] = dataframe_b['fft_normal']

# Copy the data from the pandas array into one numpy matrix
for i in range(len(dataframe_a)):
    X[i] = dataframe_a.iloc[i]['fft_feature']
for i in range(len(dataframe_b)):
    X[i+a_len] = dataframe_b.iloc[i]['fft_feature']
    
X = np.reshape(X, [examples, 1, fft_length])
#dataset = lgb.Dataset(X,label=Y[:,0])
#params = {
#    'task': 'train',
#    'boosting_type': 'gbdt',
#    'objective': 'binary',
#    'metric': {'auc'},
#    'metric_freq': 100,
#    'num_leaves': 31,
#    'learning_rate': 0.03,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.8,
#    'bagging_freq': 5,
#    'verbose': 0
#    }
#cvret = lgb.cv(params,dataset,num_boost_round=500)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model1 = fft_model()
model1.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test),class_weight={1:1,0:6})
model1.save('fft_detector.h5')
model1.summary()


#Second model
X = -np.ones([examples, pulse_length])
Y = np.zeros([examples, 1])
Y[:a_len,0] = dataframe_a['pulse_normal']
Y[-b_len:,0] = dataframe_b['pulse_normal']

# Copy the data from the pandas array into one numpy matrix
for i in range(len(dataframe_a)):
    X[i] = dataframe_a.iloc[i]['pulse_feature']
for i in range(len(dataframe_b)):
    X[i+a_len] = dataframe_b.iloc[i]['pulse_feature']
    
X = np.reshape(X, [examples, 1, pulse_length])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model2 = pulse_model()
model2.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test),class_weight={1:1,0:8})
model2.save('pulse_detector.h5')
model2.summary()
