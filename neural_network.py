class node:

	def __init__(self, value, weight):
		self.value = value
		self.weight = weight

class neural_network:

	def __init__(self):
		pass

	def set_input_layer(self, input_nodes, values, weights):

		self.input_layer = []

		for i in range(input_nodes):
			self.input_layer.append(node(values[i], weights[i]))




a = neural_network()
