import argparse

from ungm import UNGM
from simple_sim import SimpleSim
import os

import numpy as np

from progressbar import ProgressBar

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SIM_DIR = os.path.join(BASE_DIR, 'sims')



parser = argparse.ArgumentParser()

parser.add_argument('simtype', 
                    choices=['ungm', 'simple'],
                    action='store')

parser.add_argument('--num',
                    type=int,
                    default=10000)

parser.add_argument('--T',
                    type=int,
                    default=100)

parser.add_argument('--Q',
                    type=float,
                    default=10.)
parser.add_argument('--R',
                    type=float,
                    default=10.)
parser.add_argument('--x_var',
                    type=float,
                    default=10.)

args = parser.parse_args()



print("Generating {0} simulations, T = {1}, of {2}".format(args.num, args.T, args.simtype))



observations_list = []
true_list = []

bar = ProgressBar()
for i in bar(range(args.num)):
    x_0 = np.random.normal(0, args.x_var, 1)
    if args.simtype == 'ungm':
        sim = UNGM(x_0, args.Q, args.R)
    elif args.simtype == 'simple':
        sim = SimpleSim(x_0, args.Q, args.R)

    for t in range(args.T):
        sim.process_next()

    observations_list.append(np.array(sim.get_all_y()))
    true_list.append(np.array(sim.get_all_x()))


observations_dest = os.path.join(SIM_DIR, 'obs-{0}-{1}-{2}.npy'.format(args.simtype, args.num, args.T))
true_dest= os.path.join(SIM_DIR, 'true-{0}-{1}-{2}.npy'.format(args.simtype, args.num, args.T))

print("Saving observations to {0}".format(observations_dest))
print("Saving true to {0}".format(true_dest))

np.save(observations_dest, np.array(observations_list))
np.save(true_dest, np.array(true_list))
