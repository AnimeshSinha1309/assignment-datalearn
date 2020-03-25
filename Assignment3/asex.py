"""
Genetic Algorithm code to optimize unknown function.
"""

from random import random as rnd
import numpy as np
from numpy.random import randint

import client
import datagen

SECRET_KEY = 'EdQPhzkQ1CnpQ9jxCY4AH8eATTHeZm4IwEs2P1jE2xT3p8sCeE'

LOG_FILE = 'results_6.txt'  # File that keep populations and fitness
ITERATIONS = 240
POPULATION_SIZE = 10
REAL_DATA = True

OVERFIT_WEIGHTS = np.array([
    0.0,
    0.1240317450077846,
    -6.211941063144333,
    0.04933903144709126,
    0.03810848157715883,
    8.132366097133624e-05,
    -6.018769160916912e-05,
    -1.251585565299179e-07,
    3.484096383229681e-08,
    4.1614924993407104e-11,
    -6.732420176902565e-12
])
 
class Individual:
    """
    A single individual of the Genetic Ensemble
    """

    def __init__(self, number_of_genes: int = 11, default: list = OVERFIT_WEIGHTS):
        """
        Generates one Individual of the Population
        """
        self.fitness = None
        self.train_error, self.validate_error = None, None
        self.genes = np.random.normal(loc=0.0, scale=0.0005, size=(number_of_genes)) + default

    def birth(self, parent=None, mutation_count:int = 3):
        """
        Create the new genome for a child of the parents.
        :param parents1: the first parent in the couple mating
        :param parents2: the second parent in the couple mating
        :param mutation_count: number of mutated genes
        """
        if parent is not None:
            self.genes = parent.genes.copy()
        for i in range(mutation_count):
            idx = np.random.randint(len(self.genes))
            self.genes[idx] += np.random.normal(loc=0.0, scale=0.00001)
        self.genes = np.clip(self.genes, -10, 10)
        self.update_fitness()
        return self

    def update_fitness(self, real_data: bool = False,
                       fn=lambda train, val: -(train + val + 4 * abs(val - train))) -> float:
        """
        Computes the fitness of each individual by making a call to the fitness function
        :param individual: a list containing the genome of the individual
        :param real_data: if true, makes a call to the server, dummy debug data otherwise
        :param fn: function object, describes the net loss given the validation and training losses
        :returns: fitness value of the individual
        """
        if REAL_DATA:
            train_error, validation_error = client.get_errors(
                SECRET_KEY, list(self.genes))
        else:
            train_error, validation_error = datagen.get_errors(
                SECRET_KEY, self.genes)
        self.train_error, self.validate_error = train_error, validation_error
        self.fitness = fn(train=train_error, val=validation_error)
        print(self.genes, self.train_error, self.validate_error, self.fitness)
        return self.fitness

    def __lt__(self, other):
        """
        Compares two individuals based on higher fitness
        :param self: the individual comparing (left operand)
        :param other: the individual to compare against (right operand)
        :returns: boolean, True if self is weaker than other, False otherwise
        """
        return self.fitness < other.fitness

    @staticmethod
    def generate_population(number_of_individuals: int, number_of_genes: int = 11,
                            default: np.ndarray = OVERFIT_WEIGHTS) -> list:
        """
        Creates a new population of individuals
        :param number_of_individuals: number of different individuals in the populus
        :param number_of_genes: number of real values genes the individual should have
        :returns: list of the individuals, each is a list of genes
        """
        return [Individual(number_of_genes, default).birth() for iter_x in range(number_of_individuals)]

    @staticmethod
    def selection(generation: list, population_size: int = POPULATION_SIZE) -> list:
        """
        Selects members from the current generation to the next generation (survivors).
        :param generation: the current populus
        :returns: dict, array of genomes of all selected, all fitness
        """
        generation = sorted(generation, reverse=True)
        assert all([type(person) is Individual for person in generation])
        return generation[:POPULATION_SIZE]

    @staticmethod
    def mating(parents: list, n_offsprings: int = 2) -> list:
        """
        Create the new genome for n children of the parents.
        :param (parents1): the first parent in the couple mating
        :param parents2: the second parent in the couple mating
        :param n_offsprings: number of children born to each couple
        """
        offsprings = [Individual().birth(parent)
                      for _ in range(n_offsprings)
                      for parent in parents]
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
        children = Individual.mating(generation)
        generation = Individual.selection(generation + children)
        assert all([type(person) is Individual for person in generation])

        # Keep a log of things
        with open(LOG_FILE, 'a') as LOG_FILE_OBJ:
            LOG_FILE_OBJ.write(str(Individual.stats_fitness(generation))+'\n')

    with open(LOG_FILE, 'a') as LOG_FILE_OBJ:
        LOG_FILE_OBJ.write('\n')
