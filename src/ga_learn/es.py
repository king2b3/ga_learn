"""es.py
Developer: Bayley King
Date: 2-26-2022
Description: Evolutionary Strategy framework
"""
################################## Imports ###################################
import individual as indv
import random
##############################################################################
################################# Constants ##################################
##############################################################################


class ES():
    def __init__(self, test, train, mu=10, lambda_=50) -> None:
        # recommended 1/7 ratio for mu to lambda
        self.pop_size = mu
        self.off_size = lambda_
        self.pop = []
        self.offspring = []
        self.test = test
        self.train = train
        self.pop_gen()

    def pop_gen(self) -> None:
        for _ in range(self.pop_size):
            self.pop.append(indv.Network(self.train, self.test))

    def selection(self) -> None:
        """(u+l) selection type
        All 10 parents, 30 mutated, and 30 with crossover join the
          offspring pool.
        Deterministic elite replacement creates the new parent pool
        """
        self.offspring = self.pop.copy()
        # mutations
        for i in self.pop:
            ran = round((self.off_size - self.pop_size) / (2 * self.pop_size))
            for _ in range(ran):
                self.offspring.append(i.mutate())
                self.offspring.append(i.crossover(random.choice(self.pop)))
        # calculate fitness of the offspring
        for x in self.offspring:
            x.calc_fitness()
        self.offspring.sort(key=lambda x: x.fitness, reverse=True)
        self.pop = self.offspring[:self.pop_size]

    def sort_pop(self) -> None:
        self.pop.sort(key=lambda x: x.fitness, reverse=True)

    def stats(self, g) -> str:
        """Returns dictionary with stats of the population"""
        self.sort_pop()
        temp = f"Generation: {g} "
        temp += f"Best performing: {self.pop[0].fitness} "
        temp += f"Average individual: {sum(x.fitness for x in self.pop)/self.pop_size}"
        return temp

    def exit_cond(self) -> bool:
        """Checks to see if Accuracy, Recall, and Percision are all 1"""
        # population already sorted from stats function call
        if self.pop[0].fitness == 1:
            return True
        return False

    def run(self, gens=100) -> None:
        for g in range(gens):
            self.selection()
            print(self.stats(g))
            if self.exit_cond():
                break
