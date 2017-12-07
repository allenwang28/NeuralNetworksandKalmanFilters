from ungm import UNGM
from ukf import UKF
from ekf import EKF

import keras
from keras.models import load_model

import numpy as np
from scoring import MSE

import matplotlib.pyplot as plt

from progressbar import ProgressBar


num_sims = 1000 
T = 50

first_x_0 = 0
R = 5.
Q = 1.
x_var = 0.1

training_size = 1000

lstm10 = load_model('models/lstm-units10-simungm-{0}-{1}.h5'.format(training_size, T))
lstm100 = load_model('models/lstm-units100-simungm-{0}-{1}.h5'.format(training_size, T))
rnn10 = load_model('models/rnn-units10-simungm-{0}-{1}.h5'.format(training_size, T))
rnn100 = load_model('models/rnn-units100-simungm-{0}-{1}.h5'.format(training_size, T))

all_y = []
all_x = []

ukf_mses = []
ekf_mses = []
lstm_mses = []
lstm_stacked_mses = []
bar = ProgressBar()
for s in bar(range(num_sims)):
    #x_0 = np.random.normal(0, x_var, 1)
    x_0 = np.array([0])
    sim = UNGM(x_0, R, Q, x_var)
    ukf = UKF(sim.f, sim.F,
              sim.h, sim.H,
              sim.Q, sim.R,
              5.,
              first_x_0, 1)
    ekf = EKF(sim.f, sim.F,
              sim.h, sim.H,
              sim.Q, sim.R,
              first_x_0, 1)
    for t in range(T):
        x, y = sim.process_next()
        ukf.predict()
        ukf.update(y)
        ekf.predict()
        ekf.update(y)
    ukf_mses.append(MSE(ukf.get_all_predictions(), sim.get_all_x()))
    ekf_mses.append(MSE(ekf.get_all_predictions(), sim.get_all_x()))

    all_x.append(np.array(sim.get_all_x()))
    all_y.append(np.array(sim.get_all_y()))

X = np.array(all_y)[:,:-1,:]
y = np.array(all_x)[:,1:,:]

lstm10_pred = lstm10.predict(X)
lstm100_pred = lstm100.predict(X)
rnn10_pred = rnn10.predict(X)
rnn100_pred = rnn100.predict(X)

lstm10_mses = []
lstm100_mses = []
rnn10_mses = []
rnn100_mses = []
for lstm10p, lstm100p, rnn10p, rnn100p, y_sim in zip(lstm10_pred, lstm100_pred, rnn10_pred, rnn100_pred, y):
    lstm10_mses.append(MSE(lstm10p, y_sim))
    lstm100_mses.append(MSE(lstm100p, y_sim))
    rnn10_mses.append(MSE(rnn10p, y_sim))
    rnn100_mses.append(MSE(rnn100p, y_sim))
    

print ("Number of simulations: {0}".format(num_sims))
print ("Length of simulations: {0}".format(T))

print ("---------------")
print ("UKF")
print ("{0} +- {1}".format(np.mean(ukf_mses), np.std(ukf_mses)))

print ("---------------")
print ("EKF")
print ("{0} +- {1}".format(np.mean(ekf_mses), np.std(ekf_mses)))

print ("---------------")
print ("LSTM10")
print ("{0} +- {1}".format(np.mean(lstm10_mses), np.std(lstm10_mses)))

print ("---------------")
print ("LSTM100")
print ("{0} +- {1}".format(np.mean(lstm100_mses), np.std(lstm100_mses)))

print ("---------------")
print ("RNN10")
print ("{0} +- {1}".format(np.mean(rnn10_mses), np.std(rnn10_mses)))

print ("---------------")
print ("RNN100")
print ("{0} +- {1}".format(np.mean(rnn100_mses), np.std(rnn100_mses)))

plt.plot(range(T)[1:], y_sim.reshape(y_sim.shape[0], -1), label='True', color='black')
plt.plot(range(T)[1:], ekf.get_all_predictions()[1:], label='ekf')
plt.plot(range(T)[1:], ukf.get_all_predictions()[1:], label='ukf')
plt.plot(range(T-1), lstm10p, label='lstm10')
plt.plot(range(T-1), lstm100p, label='lstm100')
plt.plot(range(T-1), rnn10p, label='rnn10')
plt.plot(range(T-1), rnn100p, label='rnn100')

plt.legend()

plt.show()



 
