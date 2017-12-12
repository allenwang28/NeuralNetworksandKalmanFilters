from robotcar import robotcar
from kalman_filter import kalman_filter
from random import seed, uniform
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import load_model

training_size = 100000
T = 150

lstm = load_model('models/lstm-units500-simrobot-{0}-{1}.h5'.format(training_size, T))

# Create Random Physical Landmarks
num_landmarks = 5
landmark_range = 300
landmarksx = np.empty(num_landmarks)
landmarksy = np.empty(num_landmarks)
seed()
for i in range(num_landmarks):
    landmarksx[i] = uniform(-landmark_range,landmark_range)
    landmarksy[i] = uniform(-landmark_range,landmark_range)


# Time Period
time = 15 # sec
ts = 0.1
n = int(time/ts)

num_features = 5 + num_landmarks * 4

e_kf = []
e_lstm = []

from progressbar import ProgressBar
bar = ProgressBar()

for sim in bar(range(500)):
    # Initialize Car and Kalman Filter
    robot = robotcar(2, 0.5, num_landmarks, ts=ts)
    kf = kalman_filter(robot)

    # Initialize empty arrays for plotting
    t = np.empty(n)
    e = np.empty(n)
    x = np.empty(n)
    y = np.empty(n)
    x_pred = np.empty(n)
    y_pred = np.empty(n)
    x_update = np.empty(n)
    y_update = np.empty(n)
    observation = np.empty((n, num_features))


    for i in range(n):
        # Create random movement of wheels
        l_wheel = uniform(5, 15)
        r_wheel = uniform(5, 15)
        robot.move_wheels(l_wheel, r_wheel)

        # Prediction Step
        kf.predict()

        # Add data to plot array
        x_pred[i] = robot.position[0,0]
        y_pred[i] = robot.position[1,0]
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
        # print "Run " + str(i) + ", updated: \n" + str(robot.position[0:3, 0])
        # print "Run " + str(i) + ", actual: \n" + str(robot.positionVector)

        # Add data to plot arrays
        x_update[i] = robot.position[0,0]
        y_update[i] = robot.position[1,0]
        x[i] = robot.positionVector[0]
        y[i] = robot.positionVector[1]
        t[i] = robot.time

    e_kf.append(np.mean(np.sqrt(np.square(x_update - x) + np.square(y_update-y))))

    observations = np.array(observation).reshape(1, n, num_features)
    lstm_pred = lstm.predict(observations)[0]
    lstm_pred_x = lstm_pred.T[0]
    lstm_pred_y = lstm_pred.T[1]
    e_lstm.append(np.mean(np.sqrt(np.square(lstm_pred_y - y) + np.square(lstm_pred_x - x))))

print ("KF RMS:")
print ("{0} +- {1}".format(np.mean(e_kf), np.std(e_kf)))
print ("LSTM RMS:")
print ("{0} +- {1}".format(np.mean(e_lstm), np.std(e_lstm)))
"""

# Plot map and error
e = np.sqrt(np.square(x_update - x) + np.square(y_update-y))
print ("Average RMS Error of Position: " + str(np.mean(e)))

observations = np.array(observation).reshape(1, n, num_features)
lstm_pred = lstm.predict(observations)[0]
lstm_pred_x = lstm_pred.T[0]
lstm_pred_y = lstm_pred.T[1]

e = np.mean(np.sqrt(np.square(lstm_pred_y - y) + np.square(lstm_pred_x - x)))
print ("Average RMS Error of Position (lstm): {0}".format(e))
"""

plt.figure(1)
actual, = plt.plot(x, y)
updated, = plt.plot(x_update, y_update)
lstm, = plt.plot(lstm_pred_x, lstm_pred_y)
lms, = plt.plot(landmarksx, landmarksy, 'o', label="Landmarks")
plt.figlegend( (actual, updated, lstm, lms), ('Actual Position', 'Updated Position', 'LSTM Predictions', 'Landmarks'), 'lower right')
plt.title('Map of Actual, Updated, and LSTM Predicted Positions')
plt.xlabel('x Position')
plt.ylabel('y Position')
plt.show()

