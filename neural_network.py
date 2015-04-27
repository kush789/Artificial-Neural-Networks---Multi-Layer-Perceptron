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
import copy

####################### Activation functions ########################

def sigmoid(x):
	return 1.0/(1.0 + numpy.exp(-x))

def sigmoid_derivative(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

########################### Node class ##############################

""" 1. input_value -> Input data 
	2. output_value -> sigmoid(input_value)
	3. weight[0] means weight from current node 
	   to first node in next layer """

class node:

	def __init__(self, input_value, bias, weight, output_value):
		self.input_value = input_value
		self.output_value = output_value
		self.bias = bias
		self.prev_weight = copy.copy(weight)
		self.weight = copy.copy(weight)

####################### Input layer class ###########################

""" List of nodes, no bias list (input has no bias input) """

class input_layer:

	def __init__(self, input_nodes, values, weights):
		self.nodes = []
		self.total = input_nodes

		for i in range(input_nodes):
			self.nodes.append(node(0, 0, weights[i], values[i]))

	def set_output_values(self, output_values):
		for i in range(self.total):
			self.nodes[i].output_value = output_values[i]

################## Hidden and Output layer class ####################

""" List of nodes, and a bias list.
	Bias[i] means bias of input to node i of current layer """

class layer:

	def __init__(self, num_nodes, weights, bias):
		self.nodes = []
		self.delta = []
		self.bias = copy.copy(bias)
		self.total = num_nodes

		for i in range(num_nodes):
			self.nodes.append(node(0, bias[i], weights[i], 0))
			self.delta.append(0)

################# Artificial Neural Network class ###################

""" As of now, only one hidden layer """

class neural_network:

	def __init__(self, learning_rate, mobility_factor):
		self.learning_rate = learning_rate
		self.mobility_factor = mobility_factor

	""" setting constants """

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def set_mobility_factor(self, mobility_factor):
		self.mobility_factor = mobility_factor

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

			temp_sum += self.hidden_layer.nodes[i].bias
			self.hidden_layer.nodes[i].input_value = temp_sum
			self.hidden_layer.nodes[i].output_value = sigmoid(temp_sum)

	def set_output_layer_values(self):

		""" Sets the values of the output layer nodes """

		for i in range(self.output_layer.total):
			temp_sum = 0.0

			for j in range(self.hidden_layer.total):
				temp_sum += (self.hidden_layer.nodes[j].output_value * 
							 self.hidden_layer.nodes[j].weight[i])

			temp_sum += self.output_layer.nodes[i].bias
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

		""" Calculates delta values for each node """

		""" Output layer nodes """

		for i in range(self.output_layer.total):
			self.output_layer.delta[i] = sigmoid_derivative(
				 self.output_layer.nodes[i].input_value) * (
				 output_data[i] - 
				 self.output_layer.nodes[i].output_value)
		
		""" Hidden layer nodes """

		for i in range(self.hidden_layer.total):
			self.hidden_layer.delta[i] = 0

			for j in range(len(self.hidden_layer.nodes[i].weight)):
				self.hidden_layer.delta[i] += (
					 self.hidden_layer.nodes[i].weight[j] *
					 self.output_layer.delta[j])

			self.hidden_layer.delta[i] *= sigmoid_derivative(
					 self.hidden_layer.nodes[i].input_value)

	def update_weights(self):

		""" Updates weights for each edge """

		""" Hidden layer nodes """

		for i in range(self.hidden_layer.total):
			current_weights = [0 for j in range(self.output_layer.total)]

			for j in range(self.output_layer.total):
				current_weights[j] = self.hidden_layer.nodes[i].weight[j]

				self.hidden_layer.nodes[i].weight[j] += ( 
					 self.mobility_factor * 
					 self.hidden_layer.nodes[i].prev_weight[j])
				self.hidden_layer.nodes[i].weight[j] += (
					 self.learning_rate * 
					 self.output_layer.delta[j] * 
					 self.hidden_layer.nodes[i].output_value)

				self.hidden_layer.nodes[i].prev_weight[j] = current_weights[j]

		""" Input layer nodes """

		for i in range(self.input_layer.total):
			current_weights = [0 for j in range(self.hidden_layer.total)]

			for j in range(self.hidden_layer.total):
				current_weights[j] = self.input_layer.nodes[i].weight[j]

				self.input_layer.nodes[i].weight[j] += ( 
					 self.mobility_factor * 
					 self.input_layer.nodes[i].prev_weight[j])
				self.input_layer.nodes[i].weight[j] += (
					 self.learning_rate * 
					 self.hidden_layer.delta[j] * 
					 self.input_layer.nodes[i].output_value)

				self.input_layer.nodes[i].prev_weight[j] = current_weights[j]

	def back_propogation(self, output_data):

		""" Complete back propogation learning """

		self.delta_calculation(output_data)
		self.update_weights()

	def learn(self, input_data, output_data):

		""" Complete supervised learning """

		self.feed_forward(input_data)
		self.back_propogation(output_data)

	""" Print values of various attributes """

	def print_weights(self):

		print "prev input layer"
		for i in range(self.input_layer.total):
			print self.input_layer.nodes[i].prev_weight,

		print "\n\ncurrent input layer"
		for i in range(self.input_layer.total):
			print self.input_layer.nodes[i].weight,


		print "\n\nprev hidden layer"
		for i in range(self.hidden_layer.total):
			print self.hidden_layer.nodes[i].prev_weight,

		print "\n\ncurrent hidden layer"
		for i in range(self.hidden_layer.total):
			print self.hidden_layer.nodes[i].weight,

		print "\n\n"

	def print_node_values(self):

		print "input layer"
		for i in range(self.input_layer.total):
			print "node", i+1, "output :", self.input_layer.nodes[i].output_value

		print "\n\nhidden layer"
		for i in range(self.hidden_layer.total):
			print "node", i+1, "input :", self.hidden_layer.nodes[i].input_value,
			print "node", i+1, "output :", self.hidden_layer.nodes[i].output_value

		print "\n\noutput layer"
		for i in range(self.output_layer.total):
			print "node", i+1, "input :", self.output_layer.nodes[i].input_value,
			print "node", i+1, "output :", self.output_layer.nodes[i].output_value

		print "\n\n"

	def print_delta_values(self):

		print "hidden layer"
		for i in range(self.hidden_layer.total):
			print "node", i+1, "delta :", self.hidden_layer.delta[i]

		print "\n\noutput layer"
		for i in range(self.output_layer.total):
			print "node", i+1, "delta :", self.output_layer.delta[i]

		print "\n\n"
