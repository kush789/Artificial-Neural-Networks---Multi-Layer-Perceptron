import numpy

#################### Activation functions ######################

def sigmoid(x):
	return 1.0/(1.0 + numpy.exp(-1))

def sigmoid_derivative(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

######################## Node class ############################

class node:

	def __init__(self, value, weight):
		self.value = value
		self.weight = weight

#################### Input layer class ##########################

""" List of nodes, no bias list (input has no bias input) """

class input_layer:

	def __init__(self, input_nodes, values, weights):
		self.nodes = []

		for i in range(input_nodes):
			self.nodes.append(node(values[i], weights[i]))

############### Hidden and Output layer class ###################

""" List of nodes, and a bias list """

class layer:

	def __init__(self, hidden_nodes, values, weights, bias):
		self.nodes = []
		self.bias = []

		for i in range(hidden_nodes):
			self.nodes.append(node(0, weights[i]))
			self.bias = bias[i]

############## Artificial Neural Network class ##################

""" As of now, only one hidden layer """

class neural_network:

	def __init__(self):
		pass

	def set_input_layer(self, input_nodes, values, weights):
		self.input_layer = input_layer(input_nodes, values, weights)

	def set_hidden_layer(self, hidden_nodes, values, weights, bias):
		self.hidden_layer = layer(hidden_nodes, values, weights, bias)

	def set_output_layer(self, output_nodes, values, weights, bias):
		self.output_layer = layer(output_nodes, values, weights, bias)






