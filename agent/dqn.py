import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

	def __init__(self, input_dim, hidden, output_dim):

		super(NeuralNetwork, self).__init__()

		fc = [input_dim] + hidden
		self.fc_layers = nn.ModuleList()

		for i in range(len(fc)-1):
			self.fc_layers.append(nn.Linear(fc[i], fc[i+1]))
		self.output_layer = nn.Linear(fc[-1], output_dim)
	
	def forward(self, X):
		# size = n * input_dim

		for fc in self.fc_layers:
			X = torch.relu(fc(X))
		X = self.output_layer(X)

		return X
