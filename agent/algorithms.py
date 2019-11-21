import sys
import math
import numpy as np

sys.path.append('./pool')
import config

allAlgos = {"random":0, "Q-Learning":1}

class Algorithms:
	def __init__(self, algo):
		self.algorithm = allAlgos[algo]

	def output(self, state, reward):
		angle = np.random.uniform(-math.pi,math.pi)
		distance = np.random.randint(config.cue_max_displacement/2,config.cue_max_displacement)
		return angle, distance