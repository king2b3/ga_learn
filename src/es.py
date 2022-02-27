"""es.py
Developer: Bayley King
Date: 2-26-2022
Descrition: Evolutionary Strategy framework
"""
################################## Imports ###################################
import individual as indv
from copy import deepcopy
import random
##############################################################################
################################# Constants ##################################
##############################################################################

class ES():
	def __init__(self, u=10, l=70) -> None:
		#recommended 1/7 ratio for mew to lambda
		self.pop_size = u
		self.off_size = l
		self.pop = []
		self.offspring = []

	def pop_gen(self) -> None:
		for _ in range(self.pop_size):
			self.pop.append(indv.Network())

	def selection(self) -> None:
		"""(u+l) selection type
		All 10 parents, 30 mutated, and 30 with crossover join the 
		  offspring pool.
		Deterministic elite replacement creates the new parent pool"""
		self.offspring = deepcopy(self.pop)
		#mutations
		for i in self.pop:
			ran = round((self.off_size - self.pop_size) / (2*self.pop_size))
			for _ in range(ran):
				self.offspring.append(i.mutate())
				self.offspring.append(i.crossover(random.choice(self.pop)))
		#calculate fitness of the offspring
		[x.calc_fitness for x in self.offspring]
		self.offspring.sort(key=lambda x: x.fitness, reverse=True)

	def sort_pop(self) -> None:
		self.pop_sort.sort(key=lambda x: x.fitness, reverse=True)

	def stats(self, g) -> dict:
		"""Returns dictionary with stats of the population"""
		self.sort_pop()
		print(f"Generation: {g}")
		print(f"Best performing: {self.pop[0].fitness}")
		print(f"Average individual: {sum(x.fitness for x in self.pop)}")

	def exit_cond(self) -> bool:
		"""Checks to see if Accuracy, Recall, and Percision are all 1"""
		#population already sorted from stats function call
		if self.pop[0].fit == 1:
			return True
		return False

	def run(self, gens=100) -> None:
		for g in range(gens):
			self.selection()
			print(self.stats(g))
			if self.exit_cond():
				break