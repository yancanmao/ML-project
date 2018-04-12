# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:47:16 2018

@author: lenovo
"""
import sys
import os
import numpy as np
import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt
from scipy.fftpack import fft
from keras.models import load_model


fft_length = 120
pulse_length = 400


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


def heartsound_classifier(name='./',modelfile1='fft_detector.h5',modelfile2='pulse_detector.h5'):
    model1 = load_model(modelfile1)
    model2 = load_model(modelfile2)
    if name.endswith('wav'):
        x = wav.read(name)
        fft_feature = np.abs(fft(x[1],2048))[:fft_length].reshape(1,1,fft_length)
        pulse_feature = get_pulse_feature(x).reshape(1,1,pulse_length)
        filenames = [name]
        n = 1
    else:
        filenames = [item for item in os.listdir(name) if item.endswith('wav')]
        if not name.endswith('/'):
            name += '/'
        n = len(filenames)
        fft_feature = np.zeros((n,1,fft_length))
        pulse_feature = np.zeros((n,1,pulse_length))
        i = 0
        for filename in filenames:
            x = wav.read(name+filename)
            fft_feature[i,0] = np.abs(fft(x[1],2048))[:fft_length]
            pulse_feature[i,0] = get_pulse_feature(x)
            i += 1
    
    murmur = 1-model1.predict(fft_feature)
    extra = 1-model2.predict(pulse_feature)
    
    print('name \t\t murmur likelihood \t extra stole likelihood')
    for i in range(n):
        print(filenames[i]+'\t%.3f\t\t\t%.3f'%(murmur[i], extra[i]))
    
if __name__=='__main__':
    if len(sys.argv) == 1:
        heartsound_classifier()
    if len(sys.argv) == 2:
        heartsound_classifier(sys.argv[1])
    if len(sys.argv) == 4:
        heartsound_classifier(sys.argv[1],sys.argv[2],sys.argv[3])
        