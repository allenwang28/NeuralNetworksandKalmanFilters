from ungm import UNGM
from ukf import UKF
from ekf import EKF

import numpy as np
from scoring import MSE

from progressbar import ProgressBar


num_sims = 500
T = 100

x_0 = 0
R = 10
Q = 10


ukf_mses = []
ekf_mses = []
bar = ProgressBar()
for s in bar(range(num_sims)):
    x_0 = np.random.normal(0, 10)
    sim = UNGM(x_0, R, Q, 1.)
    ukf = UKF(sim.f, sim.F,
              sim.h, sim.H,
              sim.Q, sim.R,
              5.,
              x_0, 1)
    ekf = EKF(sim.f, sim.F,
              sim.h, sim.H,
              sim.Q, sim.R,
              x_0, 1)
    for t in range(T):
        x, y = sim.process_next()
        ukf.predict()
        ukf.update(y)
        ekf.predict()
        ekf.update(y)
    ukf_mses.append(MSE(ukf.get_all_predictions(), sim.get_all_x()))
    ekf_mses.append(MSE(ekf.get_all_predictions(), sim.get_all_x()))


print ("Number of simulations: {0}".format(num_sims))
print ("Length of simulations: {0}".format(T))

print ("---------------")
print ("UKF average MSE")
print (np.mean(ukf_mses))

print ("---------------")
print ("EKF average MSE")
print (np.mean(ekf_mses))
 
