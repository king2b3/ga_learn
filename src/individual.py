"""es.py
Developer: Bayley King
Date: 2-26-2022
Descrition: Individual for ES
"""
################################## Imports ###################################
import os
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from copy import deepcopy
##############################################################################
################################# Constants ##################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	#mute annoying message at the beginning of the TS model
tf.random.set_seed(42)
EPOCHS = 25
ACTIVATIONS = [
	#"elu",
	#"exponential",
	"gelu",
	#"hard_sigmoid",
	#"linear",
	"relu",
	"selu",
	"sigmoid",
	"softplus",
	"softsign",
	"swish",
	"tanh"
]
LOSS = [
	tf.keras.losses.binary_crossentropy,
	tf.keras.losses.MSE,
	tf.keras.losses.MAE,
	tf.keras.losses.huber,
	tf.keras.losses.log_cosh,
	tf.keras.losses.poisson
]
MAX_NEURONS=25
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
##############################################################################
class Network():
	def __init__(self, train:dict, test:dict) -> None:
		"""Initial Parameter Ranges
		lr: (0,.2]
		number of layers: [2,10]	
		"""
		self.train = train
		self.test = test
		self.model = None
		self.repr = {
			"lr": random.uniform(0.001, 0.2),
			"num_layers": random.randrange(2,10),
			"layers": [],
			"loss": random.choice(LOSS)
		}
		for _ in range(self.repr["num_layers"]):
			self.repr["layers"].append([
				random.randint(1, MAX_NEURONS),
				random.choice(ACTIVATIONS)
			])

		#FIX THE BEGINNING AND ENDING LAYER SIZE
		self.repr["layers"][0][0] = 13
		self.repr["layers"][-1][0] = 1
		self.repr["layers"][-1][1] = "sigmoid"
		self.fitness = 0

	def build_model(self) -> None:
		temp = []
		for l in self.repr["layers"]:
			#print(l)
			temp.append(tf.keras.layers.Dense(l[0], activation=l[1]))
		self.model = tf.keras.Sequential(temp)

	def calc_fitness(self) -> None:
		"""Calculates the fitness"""
		self.build_model()
		self.model.compile(
			loss=self.repr["loss"],
			optimizer=tf.keras.optimizers.Adam(lr=self.repr["lr"]),
			metrics=[
				tf.keras.metrics.BinaryAccuracy(name='accuracy'),
				tf.keras.metrics.Precision(name='precision'),
				tf.keras.metrics.Recall(name='recall')
			]
		)
		
		self.model.fit(self.train["x"], self.train["y"], epochs=EPOCHS, verbose=0)
		predictions = self.model.predict(self.test["x"])

		prediction_classes = [
			1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
		]

		fit = 0
		fit += accuracy_score(self.test["y"], prediction_classes)
		fit += precision_score(self.test["y"], prediction_classes, zero_division=0)
		fit += recall_score(self.test["y"], prediction_classes, zero_division=0)
		self.fitness = fit/3

	def mutate(self) -> "Network":
		"""use gaussian distribution to change parameters for the network"""
		mutation = 2 + len(self.repr["layers"])
		mutation = random.randint(0,mutation)
		if mutation == 0: #lr
			self.repr["lr"] += np.random.normal(0,.02,1)[0]
		elif mutation == 1: #num_layers
			self.repr["num_layers"] += round(np.random.normal(0,1,1)[0])
			if self.repr["num_layers"] < 2:
				self.repr["num_layers"] = 2
		elif mutation == 2: #loss
			self.repr["loss"] = random.choice(LOSS)
		elif mutation == 3: #initial layer
			self.repr["layers"][0][1] = random.choice(ACTIVATIONS)
		elif mutation == 4: #final layer
			#need to look more into layers
			...
			#self.repr["layers"][-1][1] = random.choice(ACTIVATIONS)
		else: #other layers
			if random.choice([True, False]):
				#find std of uniform distro of neurons up to max neuron
				temp = list(range(1, MAX_NEURONS+1))
				std = np.std(temp)
				self.repr["layers"][mutation - 4][0] += round(np.random.normal(0,std/4,1)[0]) #std/4 found through trial and error
				#edge cases
				if self.repr["layers"][mutation - 4][0] > MAX_NEURONS:
					self.repr["layers"][mutation - 4][0] = MAX_NEURONS
				elif self.repr["layers"][mutation - 4][0] < 1:
					self.repr["layers"][mutation - 4][0] = 1
			else:
				self.repr["layers"][mutation - 4][1] = random.choice(ACTIVATIONS)
		
		return self

	def crossover(self, indv2) -> "Network":
		"""Caller func for crossover"""
		mutation = 0 + len(self.repr["layers"])
		mutation = random.randint(0, mutation)
		if mutation == 0: #lr
			self.intermediate_crossover(indv2, "lr")
		elif mutation == 1: #num_layers
			self.intermediate_crossover(indv2, "num_layers")
		elif mutation == 2: #loss
			self.discrete_crossover(indv2, "loss")
		elif mutation == 3: #initial or final layer
			self.discrete_crossover(indv2, "layers", mutation)
		else: #other layers
			if random.choice([True, False]):
				self.discrete_crossover(indv2, "layers", mutation)
			else:
				self.intermediate_crossover(indv2, "layers", mutation)

		return self

	def discrete_crossover(self, indv2, gene, mutation=None) -> None:
		"""Discrete crossover indv 1 (self) and indv 2
		Equal chance for each parents' parameter
		Used in variable params
			eg. Activation types for a layer"""
		if mutation:
			#layer choice
			indv2_mutation = random.randint(0, len(indv2.repr["layers"]) - 1)
			#print(mutation)
			self.repr[gene][mutation - 4][1] = random.choice([self.repr[gene][mutation - 4][1], 
				indv2.repr[gene][indv2_mutation][1]])
		elif random.choice([True, False]):
				self.repr[gene] = indv2.repr[gene]

	def intermediate_crossover(self, indv2, gene, mutation=None) -> None:
		"""Intermediate crossover indv 1 (self) and indv 2
		Averaged result of parents' parameter
		Used in stragetic params
			eg. Number of layers"""
		if mutation:
			#layers choice
			indv2_mutation = random.randint(0,len(indv2.repr["layers"]) - 1)
			self.repr[gene][mutation - 4][0] = round((self.repr[gene][mutation - 4][0] + 
				indv2.repr[gene][indv2_mutation][0]) / 2)
		else:
			self.repr[gene] = (self.repr[gene] + indv2.repr[gene])/2
			if gene == "num_layers":
				self.repr[gene] = round(self.repr[gene])

	def stats(self) -> None:
		"""Dump stats to a json file"""
		import json
		weights_list = self.model.get_weights()
		#print(f"number of layers {self.repr['num_layers']}")
		temp = True
		weights = []
		biases = []
		for l in range(len(weights_list)):
			if temp:
				#weights
				weights.append(list(weights_list[l]))
				temp = False
			else:
				#bias
				biases.append(list(weights_list[l]))
				temp = True

		print(biases)
		biases = json.dumps(biases)
		print(biases)
		#print(len(weights))
		import itertools
		b = list(itertools.chain.from_iterable(biases))
		w = list(itertools.chain.from_iterable(weights))
		weight_vals = list(itertools.chain.from_iterable(w))
		weight_vals = {
			"name": "weights",
			"max": max(w),
			"min": min(w)
		}
		biases_vals = {
			"names": "biases",
			"max": max(b),
			"min": min(b)
		}
		print(weight_vals, biases_vals)

		
def main():
	train = {
        "x": np.load("data-set/x_train.npy"),
        "y": np.load("data-set/y_train.npy")
    }
	test = {
        "x" : np.load("data-set/x_test.npy"), 
        "y" : np.load("data-set/y_test.npy")
    }

	algorithm = Network(train, test)
	algorithm.calc_fitness()
	algorithm.stats()

if __name__ == "__main__":
	main()