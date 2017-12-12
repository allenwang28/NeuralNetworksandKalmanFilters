# from https://github.com/Sanderi44/Kalman-Filter-Simulator/blob/master/run.py
from robotcar import robotcar
from kalman_filter import kalman_filter
from random import seed, uniform
import numpy as np
import matplotlib.pyplot as plt

from progressbar import ProgressBar
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SIM_DIR = os.path.join(BASE_DIR, 'sims')

num_sims = 100000

num_landmarks = 5
landmark_range = 300
time = 15 # sec
ts = 0.1
# Time Period
n = int(time/ts)

true = []
observations = []

bar = ProgressBar()

for sim in bar(range(num_sims)):
    # Create Random Physical Landmarks
    landmarksx = np.empty(num_landmarks)
    landmarksy = np.empty(num_landmarks)
    seed()
    for i in range(num_landmarks):
        landmarksx[i] = uniform(-landmark_range,landmark_range)
        landmarksy[i] = uniform(-landmark_range,landmark_range)

    # Initialize Car and Kalman Filter
    robot = robotcar(2, 0.5, num_landmarks, ts=ts)
    kf = kalman_filter(robot)


    # Initialize empty arrays for plotting
    t = np.empty(n)
    e = np.empty(n)
    x = np.empty(n)
    y = np.empty(n)

    # features in an observation include:
    # left_encoder
    # right_encoder
    # x_odom
    # y_odom
    # theta_odom
    # for each landmark:
    #   bearing
    #   range
    #   landmarkx
    #   landmarky
    num_features = 5 + num_landmarks * 4
    observation = np.empty((n, num_features))

    #x_pred = np.empty(n)
    #y_pred = np.empty(n)
    #x_update = np.empty(n)
    #y_update = np.empty(n)

    for i in range(n):
        # Create random movement of wheels
        l_wheel = uniform(5, 15)
        r_wheel = uniform(5, 15)
        robot.move_wheels(l_wheel, r_wheel)

        # Prediction Step
        kf.predict()

        # Add data to plot array
        #x_pred[i] = robot.position[0,0]
        #y_pred[i] = robot.position[1,0]
        observation[i][0] = robot.left_encoder
        observation[i][1] = robot.right_encoder
        observation[i][2] = robot.x_odom
        observation[i][3] = robot.y_odom
        observation[i][4] = robot.theta_odom
        # Update Steps - Perform an update to the current prediction for all landmarks
        for j in range(num_landmarks):
            landmarkx = landmarksx[j]
            landmarky = landmarksy[j]
            kf.update(landmarkx, landmarky, j)
            observation[i][5 + j*4] = robot.range
            observation[i][5 + j*4 + 1] = robot.thetaL
            observation[i][5 + j*4 + 2] = landmarkx
            observation[i][5 + j*4 + 3] = landmarky

        # Uncomment to print all position data
        # print ("Run " + str(i) + ", updated: \n" + str(robot.position[0:3, 0]))
        # print ("Run " + str(i) + ", actual: \n" + str(robot.positionVector))

        # Add data to plot arrays
        #x_update[i] = robot.position[0,0]
        #y_update[i] = robot.position[1,0]
        x[i] = robot.positionVector[0]
        y[i] = robot.positionVector[1]

        t[i] = robot.time

    observations.append(observation)
    true.append(np.array([x, y]).T)

observations = np.array(observations)
true = np.array(true)

observation_dest = os.path.join(SIM_DIR, 'obs-robot-{0}-{1}.npy'.format(num_sims, n))
true_dest = os.path.join(SIM_DIR, 'true-robot-{0}-{1}.npy'.format(num_sims, n))

print("Saving observations to {0}".format(observation_dest))
print("Saving true to {0}".format(true_dest))

np.save(observation_dest, observations)
np.save(true_dest, true)

