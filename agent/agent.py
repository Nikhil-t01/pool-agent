import sys
import math
import numpy as np

import algorithms

sys.path.append('./pool')
import config
import gamestate
import ball

class Agent:
	def __init__(self, numGrids):
		self.gameState = None
		self.state = np.zeros([16, numGrids, numGrids])
		self.reward = 0
		self.angle = None
		self.distance = None
		self.algo = algorithms.Algorithms("random")

	def returnAction(self):
		print(self.reward)
		self.stateToFeatures()
		self.getAction()
		return self.angle, self.distance

	def getReturns(self, curGameState):
		self.calculateReward(curGameState)
		self.gameState = curGameState

	def stateToFeatures(self):
		divX = math.ceil(config.resolution[0]/self.state.shape[1])
		divY = math.ceil(config.resolution[1]/self.state.shape[2])
		for y in self.gameState:
			xCoord, yCoord = math.floor(y[1][0]/divX), math.floor(y[1][1]/divY)
			self.state[y[0]][xCoord][yCoord] = 1

	def getAction(self):
		self.angle, self.distance = self.algo.output(self.state, self.reward)

	def calculateReward(self, curGameState, pref=ball.BallType.Striped):
		if self.gameState is None: # first move
			print("First Move: ")
			self.reward = 0
		elif whiteInitPos(curGameState): # white ball pot
			print("White Potted: ")
			self.reward = -1
		elif len(self.gameState) != len(curGameState):
			print("Something Potted: ")
			potted = [0]*16
			rew = 0
			for x in self.gameState:
				potted[x[0]] = 1
			for y in curGameState:
				potted[y[0]] = 0
			for x in self.gameState:
				if potted[x[0]] == 1:
					if (x[0] < 8) ^ (pref == ball.BallType.Striped): # my ball potted
						rew += 1
					else: # opponent ball potted
						rew -= 1
			self.reward = rew
		else: # nothing potted
			if notTouched(self.gameState, curGameState):
				print("Nothing Touched: ")
				self.reward = -1
			else:
				print("Touched But Not Potted: ")
				self.reward = 0

def notTouched(a, b, epsilon=0.1):
	re = [[0, 0] for i in range(16)]
	for x in b:
		re[x[0]][0] = x[1][0]
		re[x[0]][1] = x[1][1]
	for x in a:
		re[x[0]][0] -= x[1][0]
		re[x[0]][1] -= x[1][1]
	# print(re)
	return max([x[0]**2 + x[1]**2 for x in re[1:]]) < epsilon

def whiteInitPos(curGameState):
	for x in curGameState:
		if x[0] == 0:
			if x[1][0] == config.white_ball_initial_pos[0] and \
				 x[1][1] == config.white_ball_initial_pos[1]:
				return True
	return False