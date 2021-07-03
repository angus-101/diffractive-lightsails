#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:37:15 2021

@author: angusmcconnell
"""

"""
This program aims to apply the evolutionary algorithm framework (DEAP) to the 
discrete dipole approximation calculation, in order to optimise the force that
is applied to the sail.
"""

import numpy as np
from addaSeq_force_scoop import calculate_force_on_sample                      # Separate python script imported functions
from deap import creator, base, tools, algorithms
from Numpy_Deap_Tools import cxTwoPointCopy, mutFlipBitArr, init_grid          # Separate python script imported functions
from time import time
from sys import argv
from experiment_recorder import ExpRecord
from scoop import futures
import scoop

# Evolutionary algorithm paramaters that the user defines in the command line

lambda_factor = float(argv[-9])                                                # Determines the dipole density on the grid
tile_factor = int(argv[-8])                                                    # Number of sub grids per sail                                                   
grid_size = int(argv[-7])                                                      # Number of dipoles on the grid 
population_size = int(argv[-6])                                                # Population size         
num_gen = int(argv[-5])                                                        # Number of generations     
tourn_size = int(argv[-4])                                                     # Tournament size                          
cx_p = float(argv[-3])                                                         # Crossover probability   
mut_p = float(argv[-2])                                                        # Mutation probabilities 
mut_ind_p = float(argv[-1])

def eval_func(individual):
    
    """
    This function calculates the force acting on a large grid, made up of many
    smaller grids that have been tiled.
    """
    
    tiled = np.tile(individual, (tile_factor, tile_factor))
    return calculate_force_on_sample(tiled, lam_frac_=lambda_factor)

# Initialising the evolutionary algorithm

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax) 

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("attr_bool", np.random.choice, [True, False]) 
toolbox.register("individual", init_grid, creator.Individual, grid_size_= grid_size)  
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_func)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", mutFlipBitArr, indpb = mut_ind_p)
toolbox.register("select", tools.selTournament, tournsize = tourn_size)

if __name__ == "__main__":
    
    # Prints out the parameters passed to the program

    print(
        f"\n-----------------Start--------------------\n\
        Starting with parameters:\
        \nGrid Size - {grid_size}\
        \nIndividual Mutation Independent Probability - {mut_ind_p}\
        \nIndividual Mutation Probabilty - {mut_p}\
        \nCrossover Probability - {cx_p}\
        \nNumber of Generations - {num_gen}\
        \nTournament Size - {tourn_size}\
        \nPopulation - {population_size}"
    )

    s_time = time()

    pop = toolbox.population(n = population_size)                              # Setting population size in deap
    hof = tools.HallOfFame(1, similar=np.array_equal)                          # Gives the best individual in the population, np.array_equal returns true if two arrays are the same  
    
    def stat_func(ind):
        return ind.fitness.values                                              # Returns fitness values

    stats = tools.Statistics(stat_func)
    stats.register("avg", np.mean)                                             # Register average, standard deviation, minimum and maximum in the stats toolbox
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    final_pop, log = algorithms.eaSimple(                                      # Returns the final population and a logbook of the evolution
        pop,                                                                   # Population – A list of individuals.
        toolbox,                                                               # toolbox – A Toolbox that contains the evolution operators.
        cxpb = cx_p,                                                           # cxpb – The probability of mating two individuals.
        mutpb = mut_p,                                                         # mutpb – The probability of mutating an individual.
        ngen = num_gen,                                                        # ngen – The number of generation.
        stats = stats,                                                         # stats – A Statistics object that is updated inplace, optional.
        halloffame = hof,                                                      # halloffame – A HallOfFame object that will contain the best individual, optional.
        verbose = True,                                                        # verbose – Whether or not to log the statistics on the screen
    )
    
    max_force = max(log.select('max'))                                         # Selects the maximum value in logbook

    e_time = time()
    T = e_time - s_time                                                        

    print("Time taken:", T,"s, which is", T/60,'mins, and', T/3600,'hours.')   # Time of process
    
    # Locates the directory for the experiment data to be sent
    
    exp_manager = ExpRecord(
    r"C:\Users\angus\OneDrive - University of Bristol\University OneDrive\Documents\Year 4\Project\Coding\Data"   
)

    exp_manager.add_experiment(                                                # Adds the experiment with all relevant information
        hof[0],                                                                # Gives best individual to deap logbook as a .npy grid
        log,
        max_force,
        0,
        num_gen,
        population_size,
        cx_p,
        "TwoPoint",
        None,
        mut_p,
        "FlipBit",
        mut_ind_p,
        "Tournament",
        tourn_size,
    )

    exp_manager.save()                                                         # Saves data to directory
    
    
    