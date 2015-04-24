import numpy

#################### Activation functions ######################

def sigmoid(x):
	return 1.0/(1.0 + numpy.exp(-1))

def sigmoid_derivative(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

######################## Node class ############################

""" 1. Input value, list of weight of edge to each node in hidden layers 
	2. Weight[0][0] means weight from first node in current layer to 
	first node in next layer """

class node:

	def __init__(self, value, weight):
		self.value = value
		self.weight = weight

#################### Input layer class ##########################

""" List of nodes, no bias list (input has no bias input) """

class input_layer:

	def __init__(self, input_nodes, values, weights):
		self.nodes = []
		self.total = input_nodes

		for i in range(input_nodes):
			self.nodes.append(node(values[i], weights[i]))

############### Hidden and Output layer class ###################

""" List of nodes, and a bias list.
	Bias[i] means bias of input to node i of current layer """

class layer:

	def __init__(self, hidden_nodes, values, weights, bias):
		self.nodes = []
		self.bias = []
		self.total = input_nodes

		for i in range(hidden_nodes):
			self.nodes.append(node(0, weights[i]))
			self.bias.append(bias[i])

############## Artificial Neural Network class ##################

""" As of now, only one hidden layer """

class neural_network:

	def __init__(self):
		pass

	def set_input_layer(self, input_nodes, values, weights):
		self.input_layer = input_layer(input_nodes, values, weights)

	def set_hidden_layer(self, hidden_nodes, weights, bias):
		values = [0 for i in range(hidden_nodes)]
		self.hidden_layer = layer(hidden_nodes, values, weights, bias)

	def set_output_layer(self, output_nodes, bias):
		self.output_layer = layer(output_nodes, values, weights, bias)

	""" Feed Forward """

	def set_hidden_layer_values(self):

		""" Sets the values of the hidden layer """

		for i in range(self.hidden_layer.total):
			temp_sum = 0.0

			for j in range(self.input_layer.total):
				temp_sum += (self.input_layer.nodes[j].value * 
							self.input_layer.nodes[j].weights[i])

			self.hidden_layer.nodes[i].value = sigmoid(temp_sum 
											 + self.hidden_layer.bias[i])

	def set_output_layer_values(self):

		""" Sets the values of the output layer """

		for i in range(self.output_layer.total):
			temp_sum = 0.0

			for j in range(self.hidden_layer.total):
				temp_sum += (self.hidden_layer.nodes[j].value * 
							 self.input_layer.nodes[j].weights[i])

			self.output_layer.nodes[i].value = sigmoid(temp_sum 
											 + self.output_layer.bias[i])

	def feed_forward(self, input_data):

		""" Sets hidden layer values and output 
			layer values based upon input values """

		for i in range(self.input_layer.total):
			self.input_layer.nodes[i].value = input_data[i]

		set_hidden_layer_values()
		set_output_layer_values()










