#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:05:51 2020

@author: angusmcconnell
"""

"""
This program defines the tools used by DEAP so they can be imported in other
relevant programs.
"""

import numpy as np
from random import random, randint, shuffle, uniform   
from numpy.random import normal
from copy import deepcopy
import pickle as pk

def wrap_number(num):                                                          # Returns a number between 0 and 1, if number higher than 1 return 1, if number lower than 0 return 0.
    if 0 < num < 1:
        return num
    elif num < 0:
        return 0
    elif num > 1:
        return 1
    else:
        raise ValueError


def save_grid(grid, name):                                                     # Opens binary file and then saves in directory
    with open(name + ".npy", "wb") as f:
        np.save(f, grid)


def save_log(log, name):                                                       # Dumps a pickled file(f) in another log 
    with open(name + ".pkl", "wb") as f:
        pk.dump(log, f)
    

def init_grid(icls, grid_size_):
    
    """
    Generates a numpy 2d square array of arbitrary size for use 
    as an individual in the DEAP framework.

    Args:
        icls : Object used to turn normal array into individual (provided by DEAP)
        grid_size_ (int): Size of the grid (length / width)

    Returns:
        numpy 2d array: Generated individual
    """
    
    return icls(np.random.rand(grid_size_, grid_size_) > 0.5)


def init_half_grid(icls, grid_size_):
    
    """
    Generates a numpy 2d rectanglar array of size (grid_size , grid_size /2 ) for use 
    as an individual in the DEAP framework when optimsing via symmetric method.
    If grid size is odd will go up by one.

    Args:
        icls : Object used to turn normal array into individual (provided by DEAP)
        grid_size_ (int): Large length of the rectangle

    Returns:
        numpy 2d array: Generated individual
    """
    
    grid_size_even = grid_size_ if grid_size_ % 2 == 0 else grid_size_ + 1

    return icls(np.random.rand(grid_size_even, grid_size_even // 2) > 0.5)


def cxSqCopy(ind1, ind2):
    
    """
    Execute a square crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwriting in the swap operation. It prevents the 
    old list from being overwritten by the new list, and the old list remains 
    as it was and the new mutated one will be changed accordingly.  
    Taken from DEAP Tutorial Docs.
    """
    
    size = len(ind1)
    widpoint1 = randint(1, size)                                               # Collecting random starting indices for slicing (width)
    widpoint2 = randint(1, size - 1)

    lenpoint1 = randint(1, size)                                               # Collect random starting indices for slicing (length)
    lenpoint2 = randint(1, size - 1)

    if lenpoint2 >= lenpoint1:
        lenpoint2 += 1

    else:
        lenpoint1, lenpoint2 = lenpoint2, lenpoint1

    if widpoint2 >= widpoint1:
        widpoint2 += 1
    else:
        widpoint1, widpoint2 = widpoint2, widpoint1

    (
        ind1[lenpoint1:lenpoint2, widpoint1:widpoint2],
        ind2[lenpoint1:lenpoint2, widpoint1:widpoint2],
    ) = (
        ind2[lenpoint1:lenpoint2, widpoint1:widpoint2].copy(),                 # Crosses over the parts of the array within a square
        ind1[lenpoint1:lenpoint2, widpoint1:widpoint2].copy(),
    )

    return ind1, ind2


def cxTwoPointCopy(ind1, ind2):
    
    """
    Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwriting in the swap operation. It prevents the 
    old list from being overwritten by the new list, and the old list remains 
    as it was and the new mutated one will be changed accordingly.  
    """
    
    size = len(ind1)
    cxpoint1 = randint(1, size)
    cxpoint2 = randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:                                                                      # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
        ind2[cxpoint1:cxpoint2].copy(),
        ind1[cxpoint1:cxpoint2].copy(),
    )

    return ind1, ind2


def cxTwoPointCopyAd(ind1_bundle, ind2_bundle):
    
    """
    Execute a two points crossover with copy on the input individuals. 
    Designed for self adaptive algorithm. Thecopy is required because the 
    slicing in numpy returns a view of the data, which leads to a self 
    overwriting in the swap operation. It prevents the old list from being 
    overwritten by the new list, and the old list remains as it was and the 
    new mutated one will be changed accordingly.  
    DESIGNED FOR SPECIFIC SELECTION/MUTATION PARAMS
    """

    alpha = wrap_number(normal(0.5, 0.15))
    beta = wrap_number(normal(0.5, 0.15))
    delta = wrap_number(normal(0.5, 0.15))

    ind1 = ind1_bundle["grid"]
    ind2 = ind2_bundle["grid"]

    new_cxpb = alpha * ind1_bundle["cxpb"] + (1 - alpha) * ind2_bundle["cxpb"]
    new_mutpb = beta * ind1_bundle["mutpb"] + (1 - beta) * ind2_bundle["mutpb"]
    new_mutparam = (
        delta * ind1_bundle["mut_param"] + (1 - delta) * ind2_bundle["mut_param"]
    )

    size = len(ind1)
    cxpoint1 = randint(1, size)
    cxpoint2 = randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:                                                                      # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
        ind2[cxpoint1:cxpoint2].copy(),
        ind1[cxpoint1:cxpoint2].copy(),
    )

    return (
        {
            "grid": ind1,
            "cxpb": new_cxpb,
            "mutpb": new_mutpb,
            "mut_param": new_mutparam,
            "fit": None,
        },
        {
            "grid": ind2,
            "cxpb": new_cxpb,
            "mutpb": new_mutpb,
            "mut_param": new_mutparam,
            "fit": None,
        },
    )


def mutFlipBitArr(arr, indpb):
    
    """
    Performs mutFlipBit but on a 2d array

    Args:
        arr (2d numpy array): Individual grid to be mutated
        indpb (Float): Probability of a given element of being flipped
    Returns:
        2d numpy array: Mutated grid
    """
    
    for ix, iy in np.ndindex(arr.shape):
        if random() < indpb:
            arr[ix, iy] = not arr[ix, iy]
    return (arr,)


def mutFlipBitArrAd(ind_bundle):
    
    """
    Performs mutFlipBit but on a 2d array
    Designed for adaptive algorithm 

    Args:
        arr (2d numpy array): Individual grid to be mutated
        indpb (Float): Probability of a given element of being flipped
    Returns:
        2d numpy array: Mutated grid
    """
    
    arr = ind_bundle["grid"]

    for ix, iy in np.ndindex(arr.shape):
        if random() < ind_bundle["mut_param"]:
            arr[ix, iy] = not arr[ix, iy]

    new_arr = deepcopy(arr)

    new_cxpb = wrap_number(ind_bundle["cxpb"] + normal(0, 0.1))
    new_mutpb = wrap_number(ind_bundle["mutpb"] + normal(0, 0.05))
    new_mut_param = wrap_number(ind_bundle["mut_param"] + normal(0, 0.005))

    return {
        "grid": new_arr,
        "cxpb": new_cxpb,
        "mutpb": new_mutpb,
        "mut_param": new_mut_param,
        "fit": None,
    }


def selStochasticUniversalSamplingAd(individuals, k):
    
    """
    Select the *k* individuals among the input *individuals*.
    The selection is made by using a single random value to sample all of the
    individuals by choosing them at evenly spaced intervals. The list returned
    contains references to the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :return: A list of selected individuals.
    This function uses the :func:`~random.uniform` function from the python base
    :mod:`random` module.
    """
    
    s_inds = sorted(individuals, key=lambda ind: ind["fit"], reverse=True)
    sum_fits = sum(ind["fit"] for ind in individuals)

    distance = sum_fits / float(k)
    start = uniform(0, distance)
    points = [start + i * distance for i in range(k)]

    chosen = []
    for p in points:
        i = 0
        sum_ = s_inds[i]["fit"]
        while sum_ < p:
            i += 1
            sum_ += s_inds[i]["fit"]
        chosen.append(s_inds[i])

    return chosen


def varAnd(population):
    
    """
    Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation) for self adaptive algorithm individuals. 
    """
    
    offspring = [deepcopy(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    
    for i in range(1, len(offspring)):
        if i % 2 == 0 and random() < offspring[i]["cxpb"]:
            offspring[i - 1], offspring[i] = cxTwoPointCopyAd(
                offspring[i - 1], offspring[i]
            )

        if random() < offspring[i]["mutpb"]:
            offspring[i] = mutFlipBitArrAd(offspring[i])

    return offspring
