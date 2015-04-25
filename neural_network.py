#####################################################################
#																	#
# 	 This is free software; you can redistribute it and/or modify   #
#    it under the terms of the GNU General Public License as 		#
#    published by the Free Software Foundation; either version 2  	#
#    of the License, or (at your option) any later version. 		#
#																	#
#    This code is distributed in the hope that it will be useful,	#
#    but WITHOUT ANY WARRANTY; without even the implied warranty 	#
#    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 		#
#    See the GNU General Public License for more details. 			#
#																	#
#####################################################################
#																	#
#	 Copyright (C) 2015 Kushagra Singh								#
#																	#
#####################################################################

########################### Dependencies ############################

import numpy

####################### Activation functions ########################

def sigmoid(x):
	return 1.0/(1.0 + numpy.exp(-x))

def sigmoid_derivative(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

########################### Node class ##############################

""" 1. input_value -> Input data 
	2. output_value -> sigmoid(input_value)
	3. Weight[0][0] means weight from first node in current layer to 
	first node in next layer """

class node:

	def __init__(self, input_value, weight, output_value = 0):
		self.input_value = input_value
		self.output_value = output_value
		self.weight = weight

####################### Input layer class ###########################

""" List of nodes, no bias list (input has no bias input) """

class input_layer:

	def __init__(self, input_nodes, values, weights):
		self.nodes = []
		self.total = input_nodes

		for i in range(input_nodes):
			self.nodes.append(node(0, weights[i], values[i]))

	def set_output_values(self, output_values):
		for i in range(self.total):
			self.nodes[i].output_value = output_values[i]

################## Hidden and Output layer class ####################

""" List of nodes, and a bias list.
	Bias[i] means bias of input to node i of current layer """

class layer:

	def __init__(self, num_nodes, weights, bias):
		self.nodes = []
		self.bias = []
		self.delta = []
		self.total = num_nodes

		for i in range(num_nodes):
			self.nodes.append(node(0, weights[i]))
			self.bias.append(bias[i])
			self.delta.append(0)

################# Artificial Neural Network class ###################

""" As of now, only one hidden layer """

class neural_network:

	def __init__(self):
		self.learning_rate = 0.5

	""" Learning constants """

	def set_learning_rate(self, rate):
		self.learning_rate = rate

	""" Set up of layers """

	def set_input_layer(self, input_nodes, weights):
		values = [0 for i in range(input_nodes)]
		self.input_layer = input_layer(input_nodes, values, weights)

	def set_hidden_layer(self, hidden_nodes, weights, bias):
		self.hidden_layer = layer(hidden_nodes, weights, bias)

	def set_output_layer(self, output_nodes, bias):
		weights = [0 for i in range(output_nodes)]
		self.output_layer = layer(output_nodes, weights, bias)

	""" Forward Propogation """

	def set_hidden_layer_values(self):

		""" Sets the values of the hidden layer nodes """

		for i in range(self.hidden_layer.total):
			temp_sum = 0.0

			for j in range(self.input_layer.total):
				temp_sum += (self.input_layer.nodes[j].output_value * 
							self.input_layer.nodes[j].weight[i])

			temp_sum += self.hidden_layer.bias[i]
			self.hidden_layer.nodes[i].input_value = temp_sum
			self.hidden_layer.nodes[i].output_value = sigmoid(temp_sum)

	def set_output_layer_values(self):

		""" Sets the values of the output layer nodes """

		for i in range(self.output_layer.total):
			temp_sum = 0.0

			for j in range(self.hidden_layer.total):
				temp_sum += (self.hidden_layer.nodes[j].output_value * 
							 self.hidden_layer.nodes[j].weight[i])

			temp_sum += self.output_layer.bias[i]
			self.output_layer.nodes[i].input_value = temp_sum
			self.output_layer.nodes[i].output_value = sigmoid(temp_sum)

	def feed_forward(self, input_data):

		""" Sets hidden layer values and output 
			layer values based upon input values """

		self.input_layer.set_output_values(input_data)
		self.set_hidden_layer_values()
		self.set_output_layer_values()

	""" Learning through Back Propogation """

	def delta_calculation(self, output_data):

		""" Calculates the delta input_value for each 
			node in the network """