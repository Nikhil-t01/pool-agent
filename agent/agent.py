import sys
import math
import numpy as np

import algorithms

sys.path.append('./pool')
import config
import gamestate
import ball

class Agent:
	def __init__(self, numGrids=20, numAngles=20, numForces=10):
		self.gameState = None
		self.state = None
		self.action = np.zeros([numAngles,numForces])
		self.numGrids = numGrids
		self.numAngles = numAngles
		self.numForces = numForces
		self.reward = 0
		self.angle = None
		self.distance = None
		self.algo = algorithms.Algorithms("random",self.state.size,numAngles*numForces,self.state)

	def returnAction(self):
		self.getAction()
		return self.angle, self.distance

	# Sets current state and reward
	def getState(self, curGameState):
		self.calculateReward(curGameState)
		self.gameState = curGameState
		self.stateToFeatures()

	def getAction(self):
		self.action = self.algo.takeAction(self.state, self.reward)
		# self.angle = (self.action//self.numForces)*(2*math.pi)/self.numAngles - math.pi
		# self.distance = (self.action%self.numForces)*(config.cue_max_displacement/(2*self.numForces)) + config.cue_max_displacement/2
		return self.angle, self.distance

	def stateToFeatures(self):
		# divX = math.ceil(config.resolution[0]/self.state.shape[1])
		# divY = math.ceil(config.resolution[1]/self.state.shape[2])
		self.state = np.zeros([1,16*2])
		for y in self.gameState:
			# xCoord, yCoord = math.floor(y[1][0]/divX), math.floor(y[1][1]/divY)
			self.state[0][2*y[0]] = y[1][0]
			self.state[0][2*y[0]+1] = y[1][1]

	def calculateReward(self, curGameState, pref=ball.BallType.Striped):
		if self.gameState is None: # first move
			self.reward = 0
			print("First Move!")
		else:
			if self.whiteInitPos(curGameState): # white ball pot
				self.reward = -1
				print("White Ball Pot!")
			if len(self.gameState) != len(curGameState):
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
				print("Something Potted: ",rew)
				self.reward = rew
			else: # nothing potted
				if self.notTouched(self.gameState, curGameState): # Didn't touch anything
					self.reward = -1
					print("Nothing Touched")
				else: # Touched, but not potted
					self.reward = 0
					print("Touched, but not potted")

	def notTouched(self, a, b, epsilon=0.1):
		re = [[0, 0] for i in range(16)]
		for x in b:
			re[x[0]][0] = x[1][0]
			re[x[0]][1] = x[1][1]
		for x in a:
			re[x[0]][0] -= x[1][0]
			re[x[0]][1] -= x[1][1]
		return max([x[0]**2 + x[1]**2 for x in re[1:]]) < epsilon

	def whiteInitPos(self, curGameState):
		for x in curGameState:
			if x[0] == 0:
				if x[1][0] == config.white_ball_initial_pos[0] and \
					x[1][1] == config.white_ball_initial_pos[1]:
					return True
		return False