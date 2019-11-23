import sys
import math
import numpy as np
import torch

sys.path.append('./pool')
import config
import dqn

class Algorithms:
	def __init__(self, algo, numStates, numActions, initState):
		self.algorithm = algo
		# self.numAngles = numAngles
		# self.numForces = numForces
		self.numActions = numActions
		# self.numActions = numAngles*numForces
		# self.numStates = numStates
		# S
		self.state = None 
		# A
		self.action = None
		# self.w = np.zeros([numStates,numActions])
		
		if algo == "semi-gradient-sarsa":
			self.alpha = 0.5
			self.epsilon = 0.2
			self.gamma = 1.0
			self.state = initState
			# self.action = self.oneHotEncode(self.state)
		elif algo == "dqn":
			self.state = initState
			self.alpha = 0.05
			self.epsilon = 0.2
			self.nn = dqn.NeuralNetwork(initState.size, [64,128],self.numActions)
			self.action = self.epsilonGreedy(self.nn(self.state)[0].detach().numpy())

	def takeAction(self, nextState, reward):
		# algorithms should update state and action
		if self.algorithm == "random":
			self.action = self.randomAgent(nextState, reward)
		elif self.algorithm == "dqn":
			self.action = self.deepQAgent(nextState, reward)
		# elif self.algorithm == "semi-gradient-sarsa":
		# 	self.action = self.semiGradientSarsa(nextState, reward)
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
		# aprime = self.epsilonGreedy(nextState)
		# self.w[self.state][self.action] += self.alpha * (reward + self.gamma*self.qvalue(nextState,aprime) - self.qvalue(self.state,self.action)) 
		return aprime
	
	def deepQAgent(self, nextState, reward):
		with torch.no_grad():
			out_stPrime = self.nn(nextState)[0]
		out_st = self.nn(self.state)[0]
		aprime = self.epsilonGreedy(out_st.detach().numpy())

		gradient = torch.autograd.grad(out_st[self.action], self.nn.parameters())
		
		for grad, param in zip(gradient, self.nn.parameters()):
			param.data.sub_(self.alpha*(reward + out_stPrime[aprime] - out_st[self.action])*grad)
		
		return aprime
	
	# def getFeatures(self, state, action):
	# 	dimensions = self.state.shape
	# 	features = [0 for _ in range(15)]
	# 	angle = (self.action//self.numForces)*(2*math.pi)/self.numAngles - math.pi
	# 	# distance = (self.action%self.numForces)*(config.cue_max_displacement/(2*self.numForces)) + config.cue_max_displacement/2
	# 	for i in range(1,16):
	# 		displacement = self.state[i]-self.state[0]
	# 		direction = np.array([math.sin(self.angle), math.cos(self.angle)])

	# def qvalue(self, state, action):
	# 	# assuming one-hot encoded input
	# 	return self.w[state][action]

	def epsilonGreedy(self, values):
		a = None
		if np.random.random() < self.epsilon:
			a = np.argmax(values)
		else:
			a = np.random.randint(0,len(values))
		return a

	# def greedy(self, state):
	# 	# assuming feature encoding is one-hot.
	# 	return np.argmax(self.w[state])			

	# def oneHotEncode(self, idx, total):
	# 	one_hot_vector = np.zeros([total])
	# 	one_hot_vector[idx] = 1
	# 	return one_hot_vector
	