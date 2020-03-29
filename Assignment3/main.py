"""
Genetic Algorithm code to optimize unknown function.
"""

from random import random as rnd
import numpy as np
from numpy.random import randint

import client
import datagen

SECRET_KEY = 'EdQPhzkQ1CnpQ9jxCY4AH8eATTHeZm4IwEs2P1jE2xT3p8sCeE'

LOG_FILE = 'results_9.txt'  # File that keep populations and fitness
ITERATIONS = 100
POPULATION_SIZE = 40
REAL_DATA = True

ELITE_GENES = []

class Individual:
    """
    A single individual of the Genetic Ensemble
    """

    def __init__(self, default: list = [], number_of_genes: int = 11):
        """
        Generates one Individual of the Population
        """
        if len(default) != number_of_genes:
            default = ELITE_GENES[np.random.randint(len(ELITE_GENES))]
        self.fitness = 0.0
        self.genes = np.multiply(
            np.random.normal(loc=1.0, scale=0.0001, size=(number_of_genes)), 
            default)

    def birth(self, parent1 = None, parent2 = None):
        """
        Create the new genome for a child of the parents.
        :param parents1: the first parent in the couple mating
        :param parents2: the second parent in the couple mating
        """
        if parent1 is not None and parent2 is not None:
            for i in range(len(self.genes)):
                self.genes[i] = np.random.choice(
                    [parent2.genes[i], parent1.genes[i]],
                    p = [0.5, 0.5]
                )
        self.mutation()
        self.genes = np.clip(self.genes, -10, 10)
        self.update_fitness()
        return self

    def update_fitness(self, real_data: bool = False,
                fn=lambda train, val: -val) -> float:
        """
        Computes the fitness of each individual by making a call to the fitness function
        :param individual: a list containing the genome of the individual
        :param real_data: if true, makes a call to the server, dummy debug data otherwise
        :param fn: function object, describes the net loss given the validation and training losses
        :returns: fitness value of the individual
        """
        if REAL_DATA:
            train_error, validation_error = client.get_errors(SECRET_KEY, list(self.genes))
        else:
            train_error, validation_error = datagen.get_errors(SECRET_KEY, self.genes)
        print(self.genes, train_error, validation_error)
        self.fitness = fn(train=train_error, val=validation_error)
        return self.fitness

    def mutation(self, muatation_probability: float = 0.9, mutation_amount: float = 1.0):
        """
        Randomly mutates the genome.
        :param individual: individual to be mutated (or not)
        :param mutation_probability: probability of mutation
        :param mutation_amount: standard deviation of gaussian mutation
        :returns: the new mutated genome of the individual
        """
        if np.random.random() < muatation_probability:
            self.genes[np.random.randint(len(self.genes))] = np.random.random() * mutation_amount

    def __lt__(self, other):
        """
        Compares two individuals based on higher fitness
        :param self: the individual comparing (left operand)
        :param other: the individual to compare against (right operand)
        :returns: boolean, True if self is weaker than other, False otherwise
        """
        return self.fitness < other.fitness

    @staticmethod
    def generate_population(number_of_individuals: int, number_of_genes: int = 11) -> list:
        """
        Creates a new population of individuals
        :param number_of_individuals: number of different individuals in the populus
        :returns: list of the individuals, each is a list of genes
        """
        with open('answer.txt', 'r') as f:
            for st in f.readlines():
                data = st.split(']')[0].split('[')[1].strip().split(',')
                data = list(map(lambda x: float(x.strip()), data))
                ELITE_GENES.append(data)
        
        return [Individual().birth() 
                for iter_x in range(number_of_individuals)]

    @staticmethod
    def selection(generation: list, population_size: int = POPULATION_SIZE) -> list:
        """
        Selects members from the current generation to the next generation (survivors).
        :param generation: the current populus
        :returns: dict, array of genomes of all selected, all fitness
        """
        generation = sorted(generation, key = lambda x: x.fitness, reverse=True)
        assert generation[0].fitness >= generation[1].fitness >= generation[-1].fitness
        assert all([type(person) is Individual for person in generation])
        return generation[:POPULATION_SIZE]

    @staticmethod
    def pairing(generation: list, elite_fraction: float = 0.5) -> list:
        """
        Pairs up the individuals, preparing them to mate
        :param elite: a list of the elite populus
        :param selected: a complete list of the selected populus
        :returns: pairs of individuals who will mate
        """
        assert len(generation) % 2 == 0
        elite_count = int(len(generation) * 0.4)
        elites, commoners = generation[:elite_count], generation[elite_count:]        
        np.random.shuffle(elites)
        np.random.shuffle(commoners)
        generation = elites + commoners
        couples = list(zip(generation[:len(generation) // 2], generation[len(generation) // 2:]))
        assert all([type(couple) is tuple and len(couple) == 2 and type(couple[0]) is Individual 
                    and type(couple[1]) is Individual for couple in couples])
        return couples

    @staticmethod
    def mating(parents: list, n_offsprings: int = 3) -> list:
        """
        Create the new genome for n children of the parents.
        :param (parents1): the first parent in the couple mating
        :param parents2: the second parent in the couple mating
        :param n_offsprings: number of children born to each couple
        """
        offsprings = [Individual().birth(couple[0], couple[1]) 
                        for _ in range(n_offsprings) 
                        for couple in parents]
        assert all([type(person) is Individual for person in offsprings])
        return offsprings

    @staticmethod
    def stats_fitness(generation: list) -> tuple:
        """
        Gets the average and max fitness of a generation
        :param generation: the list of individuals
        :returns: (avg fitness, max_fitness)
        """
        assert all([type(person) is Individual for person in generation])
        val_avg, val_max = 0.0, -1e100
        for person in generation:
            val_avg += person.fitness
            val_max = max(val_max, person.fitness)
        return (val_avg, val_max)


if __name__ == "__main__":
    if REAL_DATA:
        CHOICE = input(("You are about to run the algorithm with {0} population and {1} " +
                        "iterations. This will take {2} calls to the server in total." +
                        "\nDo you wish to continue (y/N): ")
                       .format(POPULATION_SIZE, ITERATIONS, POPULATION_SIZE * (ITERATIONS + 1) * 2))
        if CHOICE not in ('y', 'Y'):
            exit(0)

    generation = Individual.generate_population(POPULATION_SIZE)
    assert all([type(person) is Individual for person in generation])
    
    # Keep a log of things
    with open(LOG_FILE, 'a') as LOG_FILE_OBJ:
        LOG_FILE_OBJ.write(str(Individual.stats_fitness(generation))+'\n')

    for i in range(ITERATIONS):
        print('=========================================')
        couples = Individual.pairing(generation)
        children = Individual.mating(couples)
        generation = Individual.selection(generation + children)
        assert all([type(person) is Individual for person in generation])

        # Keep a log of things
        with open(LOG_FILE, 'a') as LOG_FILE_OBJ:
            LOG_FILE_OBJ.write(str(Individual.stats_fitness(generation))+'\n')
    
    with open(LOG_FILE, 'a') as LOG_FILE_OBJ:
        LOG_FILE_OBJ.write('\n')
