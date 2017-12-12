# from https://github.com/Sanderi44/Kalman-Filter-Simulator/blob/master/kalman_filter.py
import numpy as np
from math import sin, cos 

class kalman_filter():
	def __init__(self, robot):
		self.robot = robot

	def predict(self):
		""" Predicts the new states and covariance matrix based on the system process_model """

		# Gets odometry for current move (velocity of robot and rotational velocity or robot)
		self.robot.get_odometry()

		# Updates matricies A, G, P
		self.robot.update_prediction_matrices()
		
		# Predicts position based on old position and odometry readings that are input into the system model
		self.robot.position = self.robot.position + self.robot.process_model

		# Creates Prediction Propogation matrix
		self.robot.P = np.dot(self.robot.A, np.dot(self.robot.P, self.robot.A.T)) + np.dot(self.robot.G, np.dot(self.robot.Q, self.robot.G.T))

	def update(self, landmarkx, landmarky, landmark_num):
		""" Updates the current prediction by comparing with movement relative to a landmark """

		# Gets range information for current landmark
		self.robot.get_ranger(landmarkx, landmarky)

		# Updates correction matries H, h, R, Z, V
		self.robot.update_correction_matrices(landmarkx, landmarky, landmark_num)

		# Computes difference between predicted range values and detected range values
		Y = self.robot.Z - self.robot.h

		# Computes kalman gain
		kalman_gain = np.dot(np.dot(self.robot.P, self.robot.H.T), np.linalg.inv(np.dot(self.robot.H, np.dot(self.robot.P, self.robot.H.T)) + np.dot(self.robot.V, np.dot(self.robot.R, self.robot.V.T))))
		
		# Updates states with correction based on kalman gain and ranger values
		self.robot.position = self.robot.position + np.dot(kalman_gain, Y)

		# Updates Propogation matrix for this landmark based on kalman gain
		self.robot.P = np.dot(np.identity(3+2*self.robot.num_landmarks) - np.dot(kalman_gain, self.robot.H), self.robot.P)
		

