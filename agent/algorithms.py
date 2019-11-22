import sys
import math
import numpy as np

sys.path.append('./pool')
import config


class Algorithms:
	def __init__(self, algo, numStates, numActions, initState=0):
		self.algorithm = algo
		self.numActions = numActions
		self.numStates = numStates
		# S
		self.state = None 
		# A
		self.action = None
		self.w = np.zeros([numStates,numActions])
		
		if algo == "semi-gradient-sarsa":
			self.alpha = 0.5
			self.epsilon = 0.2
			self.gamma = 1.0
			self.state = initState
			self.action = self.oneHotEncode(self.state)


	def takeAction(self, nextState, reward):
		# algorithms should update state and action
		if self.algorithm == "random":
			self.action = self.randomAgent(nextState, reward)
		elif self.algorithm == "semi-gradient-sarsa":
			self.action = self.semiGradientSarsa(nextState, reward)
		self.state = nextState
		return self.action

	def randomAgent(self, nextState, reward):	
		# angle = np.random.uniform(-math.pi,math.pi)
		# distance = np.random.randint(config.cue_max_displacement/2,config.cue_max_displacement)
		return np.random.randint(0,self.numActions)

	def semiGradientSarsa(self, nextState, reward):
		# Some computation
		# if nextState is terminal:
		# 	self.w += self.alpha*(reward - )
		aprime = self.epsilonGreedy(nextState)
		# self.w[self.state][self.action] += self.alpha * (reward + self.gamma*self.qvalue(nextState,aprime) - self.qvalue(self.state,self.action)) 
		return aprime

	def qvalue(self, state, action):
		# assuming one-hot encoded input
		return self.w[state][action]

	def epsilonGreedy(self, state):
		a = None
		if np.random.random() < self.epsilon:
			a = self.greedy(state)
		else:
			a = np.random.randint(0,numActions)
		return a

	def greedy(self, state):
		# assuming feature encoding is one-hot.
		return np.argmax(self.w[state])			

	def oneHotEncode(self, idx, total):
		one_hot_vector = np.zeros([total])
		one_hot_vector[idx] = 1
		return one_hot_vector
	