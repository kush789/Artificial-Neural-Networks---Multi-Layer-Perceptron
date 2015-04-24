class node:

	def __init__(self, value, weight):
		self.value = value
		self.weight = weight

class input_layer:

	def __init__(self, input_nodes, values, weights):
		self.nodes = []

		for i in range(input_nodes):
			self.nodes.append(node(values[i], weights[i]))

class layer:

	def __init__(self, hidden_nodes, values, weights, bias):
		self.nodes = []
		self.bias = []

		for i in range(hidden_nodes):
			self.nodes.append(node(0, weights[i]))
			self.bias = bias[i]

class neural_network:

	def __init__(self):
		pass

	def set_input_layer(self, input_nodes, values, weights):
		self.input_layer = input_layer(input_nodes, values, weights)

	def set_hidden_layer(self, hidden_nodes, values, weights, bias):
		self.hidden_layer = layer(hidden_nodes, values, weights, bias)

	def set_output_layer(self, output_nodes, values, weights, bias):
		self.output_layer = layer(output_nodes, values, weights, bias)






