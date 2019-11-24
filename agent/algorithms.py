import sys
import math
import numpy as np
import torch

sys.path.append('./pool')
import config
import dqn

class Algorithms:
	def __init__(self, algo, numStates, numActions, initState, numAngles=20, numForces=5, train=True):
		self.algorithm = algo
		self.numAngles = numAngles
		self.numForces = numForces
		self.numActions = numAngles*numForces
		# self.numActions = numAngles*numForces
		# self.numStates = numStates
		self.epsilon = 0
		# S
		self.state = None 
		# A
		self.action = None
		
		if algo == "semi-grad-sarsa":
			self.alpha = 0.5
			self.epsilon = 0.5
			if train == False:
				self.epsilon = 0
			self.gamma = 1.0
			self.state = initState
			self.numFeatures = 16
			self.action = np.random.randint(0,self.numActions)
			self.w = np.random.randn(self.numFeatures)
			# print(self.state)
			# self.action = self.oneHotEncode(self.state)
		elif algo == "dqn":
			self.state = self.normalize(initState)
			# print(self.state.shape)
			self.alpha = 0.05
			self.epsilon = 0.3
			if train == False:
				self.epsilon = 0
			self.nn = dqn.NeuralNetwork(self.state.size, [64,128],self.numActions)
			self.action = self.epsilonGreedy(self.nn(self.state)[0].detach().numpy())

	def saveModel(self):
		if self.algorithm == "dqn":
			torch.save(self.nn, "dqnModel")

	def takeAction(self, nextState, reward):
		# algorithms should update state and action
		if self.algorithm == "random":
			self.action = self.randomAgent(nextState, reward)
		elif self.algorithm == "dqn":
			nextState = self.normalize(nextState)
			self.action = self.deepQAgent(nextState, reward)
		elif self.algorithm == "semi-grad-sarsa":
			self.action = self.semiGradientSarsa(nextState, reward)
		elif self.algorithm == "closest-greedy":
			self.action = self.closestGreedy(nextState, reward)
		self.epsilon -= self.epsilon/1000
		# elif self.algorithm == "semi-gradient-sarsa":
		# 	self.action = self.semiGradientSarsa(nextState, reward)
		self.state = nextState
		return self.action

	def randomAgent(self, nextState, reward):	
		angle = np.random.uniform(-math.pi,math.pi)
		distance = np.random.randint(config.cue_max_displacement/2,config.cue_max_displacement)
		return angle, distance

	def closestGreedy(self, nextState, reward):
		numBalls = 7
		# Potted balls to be ignored
		distances = np.zeros([numBalls])
		flag = False
		for i in range(numBalls):
			j = i+9
			if nextState[j][2] == 0:
				distances[i] = math.inf
			else:	
				flag = True
				diff = np.array([nextState[j][0]-nextState[0][0], nextState[j][1]-nextState[0][1]])
				distances[i] = diff@diff
		# print(distances)
		if flag:
			idx = 9+np.argmin(distances)
		else:
			idx = 8
		# print(idx)
		displacement = np.array([nextState[idx][0]-nextState[0][0], nextState[idx][1]-nextState[0][1]])
		displacement = displacement/np.linalg.norm(displacement)
		# print(displacement)
		angle = math.asin(displacement[0])
		if displacement[1] > 0:
			angle = math.pi - angle
		angle = 2*math.pi - angle
		# print(angle)
		distance = np.random.randint(config.cue_max_displacement/2,config.cue_max_displacement)
		return angle, distance

	def semiGradientSarsa(self, nextState, reward):
		# Some computation
		# if nextState is terminal:
		# 	self.w += self.alpha*(reward - )
		# aprime = self.epsilonGreedy(nextState)
		# self.w[self.state][self.action] += self.alpha * (reward + self.gamma*self.qvalue(nextState,aprime) - self.qvalue(self.state,self.action)) 
		X_s = self.getStateFeatures(self.state,self.action)

		values = []
		for action_num in range(self.numActions):
			values.append(self.w@self.getStateFeatures(nextState, action_num))
		aprime = self.epsilonGreedy(values)
		X_sprime = self.getStateFeatures(nextState, aprime)

		self.w += self.alpha*(reward + self.w@X_sprime - self.w@X_s)*X_s
		return aprime
	
	def getStateFeatures(self, curState, action_num):
		angle = (self.action//self.numForces)*(2*math.pi)/self.numAngles - math.pi
		direction = np.array([math.sin(angle), math.cos(angle)])
		features = np.zeros(curState.shape[0])
		for i in range(1,curState.shape[0]):
			if curState[i][2] == 0:
				continue
			if direction@np.array([curState[i][0]-curState[0][0], curState[i][1]-curState[0][1]]) > 0:
				features[i] = 1
		return features

	def deepQAgent(self, nextState, reward):
		with torch.no_grad():
			out_stPrime = self.nn(nextState)[0]
		out_st = self.nn(self.state)[0]
		aprime = self.epsilonGreedy(out_st.detach().numpy())

		gradient = torch.autograd.grad(out_st[self.action], self.nn.parameters())
		
		for grad, param in zip(gradient, self.nn.parameters()):
			param.data.sub_(self.alpha*(reward + out_stPrime[aprime] - out_st[self.action])*grad)
		
		return aprime

	def normalize(self, curState):
		state = np.zeros(curState.shape)
		for i in range(curState.shape[0]):
			state[i][0] = 2*curState[i][0]/config.resolution[0] - 1
			state[i][1] = 2*curState[i][1]/config.resolution[1] - 1
			state[i][2] = curState[i][2]
		return state.reshape(1,-1)
	
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
			a = np.random.randint(0,len(values))
		else:
			a = int(np.argmax(values))
			# print(type(a),a,values[a])
		return a

	# def greedy(self, state):
	# 	# assuming feature encoding is one-hot.
	# 	return np.argmax(self.w[state])			

	# def oneHotEncode(self, idx, total):
	# 	one_hot_vector = np.zeros([total])
	# 	one_hot_vector[idx] = 1
	# 	return one_hot_vector
	