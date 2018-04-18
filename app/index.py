from bottle import *
import pandas as pd
import numpy as np
from model import *
# from heart_sound_verify import *
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import os
from keras.models import load_model


@route('/css/<filename:path>')
def server_static(filename):
    return static_file(filename, root='./css/')


@route('/js/<filename:path>')
def server_static(filename):
    return static_file(filename, root='./js/')


@route('/dataset1')
def index():
    return template('index')


@route('/dataset3')
def index():
    return template('input')


@post('/dataset1')
def index():
    age = float(request.POST.getunicode('age'))
    sex = float(request.POST.getunicode('sex'))
    cp = float(request.POST.getunicode('cp'))
    restbp = float(request.POST.getunicode('restbp'))
    chol = float(request.POST.getunicode('chol'))
    fbs = float(request.POST.getunicode('fbs'))
    restecg = float(request.POST.getunicode('restecg'))
    thalach = float(request.POST.getunicode('thalach'))
    exang = float(request.POST.getunicode('exang'))
    oldpeak = float(request.POST.getunicode('oldpeak'))
    slope = float(request.POST.getunicode('slope'))
    ca = float(request.POST.getunicode('ca'))
    thal = float(request.POST.getunicode('thal'))
    # num = request.POST.getunicode('num')
    model = request.POST.getunicode('model')
    data = [age, sex, cp, restbp, chol, fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal]
    if model == "logistic":
        return predict(logistic_regression(data))
    elif model == "naive":
        return predict(naive_bayes(data))
    elif model == "svm":
        return predict(SVM(data))
    else:
        return predict(True)
    # return "hhh"


@post('/dataset3')
def index():
    for i in request.files:
        print(i)
    wav_file = request.files.get("input-b2")
    save_path = "./dataset3/"
    file_path = save_path+wav_file.filename
    if os.path.exists(file_path):
        os.remove(file_path)
    wav_file.save(file_path)
    
    #begin to process
    '''
    fft_length = 120
    data = data_process(save_path, wav_file.filename)
    X = -np.ones([1, fft_length])
    X[0] = data.iloc[0]['fft_feature']
    X = np.reshape(X, [1, 1, fft_length])
    model1 = fft_model()
    model1.load_weights("./fft_detector.h5")
    result1 = model1.predict(X)
    print(result1[0][0])
    '''
    print (file_path)
    murmur, extra = heartsound_classifier(file_path)
    print(murmur, extra)
    
    msg = ""
    if murmur[0][0]>extra[0][0]:
        if murmur[0][0]>0.8:
            msg = "You have a high risk of murmur heart"
        else:
            msg = "You may not have a heart disease"
    elif extra[0][0]>0.8:
        msg = "You have a risk of extrahls heart"
    else:
        msg = "You may not have a heart disease"
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"error": msg}


def heartsound_classifier(name,modelfile1='./fft_detector.h5',modelfile2='./pulse_detector.h5'):
    model1 = load_model(modelfile1)
    model2 = load_model(modelfile2)
    # fft_feature = None
    # pulse_feature = None
    fft_length = 120
    pulse_length = 400
    
    x = wav.read(name)
    fft_feature = np.abs(fft(x[1],2048))[:fft_length].reshape(1,1,fft_length)
    pulse_feature = get_pulse_feature(x).reshape(1,1,pulse_length)
    filenames = [name]
    n = 1
    
    murmur = 1-model1.predict(fft_feature)
    extra = 1-model2.predict(pulse_feature)
    
    return murmur, extra
        

def get_pulse_feature(x,pulse_length=400):
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


def predict(result):
    if result is True:
        return "It's probably a heart disease"
    else:
        return "probably no heart disease"


def xgboost(input):
    return True


run(host='0.0.0.0', port=8080, debug=True)


