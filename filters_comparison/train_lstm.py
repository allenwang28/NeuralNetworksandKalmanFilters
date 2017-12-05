import keras
import pandas as pd
import numpy as np
import os

import argparse

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()

parser.add_argument('simtype', 
                    choices=['ungm'],
                    action='store')

parser.add_argument('--num',
                    type=int,
                    default=10000)

parser.add_argument('--T',
                    type=int,
                    default=100)

parser.add_argument('--units',
                    type=int,
                    default=100)

parser.add_argument('--epochs',
                    type=int,
                    default=1000)

args = parser.parse_args()




BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SIM_DIR = os.path.join(BASE_DIR, 'sims')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

TRUE_PATH = os.path.join(SIM_DIR, 'true-{0}-{1}-{2}.csv'.format(args.simtype, args.num, args.T))
OBS_PATH = os.path.join(SIM_DIR, 'obs-{0}-{1}-{2}.csv'.format(args.simtype, args.num, args.T))
MODEL_PATH = os.path.join(MODEL_DIR, '{0}-{1}-{2}-{3}lstm.h5'.format(args.simtype, args.num, args.T, args.units))




true_df = pd.read_csv(TRUE_PATH)
obs_df = pd.read_csv(OBS_PATH)


X = obs_df.values[:,:-1]
y = true_df.values[:,1:]

X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = Sequential()
    model.add(LSTM(args.units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(y_train.shape[1], activation='linear'))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

model.fit(X_train, y_train, epochs=args.epochs, batch_size=64, validation_split = 0.33)

scores = model.evaluate(X_test, y_test, verbose=0)


model.save(MODEL_PATH)

