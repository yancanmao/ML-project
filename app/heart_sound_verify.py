import os
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Masking
from keras.optimizers import Adam

fft_length = 120
pulse_length = 400


def data_process(location, filename):
    if not location.endswith('/'):
        location += '/'
    d = {'file_name': [location+filename]}
    dataframe = pd.DataFrame(data=d)
    # dataframe['file_name'] = location+filename
    dataframe['normal'] = 0
    dataframe['fft_normal'] = 0
    dataframe['pulse_normal'] = 0
    dataframe['raw_read'] = dataframe['file_name'].apply(wav.read)
    dataframe['sample_frequency'] = dataframe['raw_read'].apply(lambda x: x[0])
    dataframe['wav_file_data'] = dataframe['raw_read'].apply(lambda x: x[1])
    dataframe['data_length'] = dataframe['raw_read'].apply(lambda x: len(x[1]))
    dataframe['fft_feature'] = dataframe['wav_file_data'].apply(
        lambda x: np.abs(fft(x, 2048))[:fft_length] / np.max(np.abs(fft(x, 2048))))
    dataframe['pulse_feature'] = dataframe['raw_read'].apply(get_pulse_feature)
    dataframe['seconds'] = dataframe['data_length'] / dataframe['sample_frequency']
    return dataframe


def get_pulse_feature(x, pulse_length=pulse_length):
    fs = x[0]
    x_abs = np.abs(x[1])
    # a heart beat pulse is shorter than 0.075s.
    pulse_feature = -np.ones(pulse_length)
    valid_pos = np.where(x_abs > -1)[0]
    valid_pos = (np.floor(valid_pos / (fs * 0.075))).astype(int)
    pulse_feature[valid_pos] = 0
    pulse_pos = np.where(x_abs > np.minimum(np.max(x_abs) * 0.4, np.median(x_abs) * 15))[0]
    pulse_pos = (np.floor(pulse_pos / (fs * 0.075))).astype(int)
    pulse_feature[pulse_pos] = 1

    return pulse_feature


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
