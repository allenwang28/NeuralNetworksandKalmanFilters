import keras
import pandas as pd
import numpy as np
import os

import argparse

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, TimeDistributed

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()

parser.add_argument('simtype', 
                    choices=['ungm', 'simple'],
                    action='store')

parser.add_argument('--num',
                    type=int,
                    default=50000)

parser.add_argument('--T',
                    type=int,
                    default=100)

parser.add_argument('--epochs',
                    type=int,
                    default=1000)

args = parser.parse_args()



BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SIM_DIR = os.path.join(BASE_DIR, 'sims')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# How many LSTM units you want in a model
# Create this number of models each time.
MODEL_UNITS = [10, 100]

sim_name = '{0}-{1}-{2}'.format(args.simtype, args.num, args.T)
TRUE_PATH = os.path.join(SIM_DIR, 'true-{0}.npy'.format(sim_name))
OBS_PATH = os.path.join(SIM_DIR, 'obs-{0}.npy'.format(sim_name))



X = np.load(OBS_PATH)[:,:-1,:]
y = np.load(TRUE_PATH)[:,1:,:]


for model_units in MODEL_UNITS:
    MODEL_PATH = os.path.join(MODEL_DIR, 'lstm-units{0}-sim{1}.h5'.format(model_units, sim_name))

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = Sequential()
        model.add(LSTM(model_units, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(y.shape[2], activation='linear')))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mse'])

    model.fit(X, y, epochs=args.epochs, batch_size=64, validation_split = 0.33)
    model.save(MODEL_PATH)

