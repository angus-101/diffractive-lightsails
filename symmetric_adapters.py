from numpy import concatenate, flip
import numpy as np

"""
This program implements the ability to reflect a grid to obtain symmetry.
We are testing whether symmetry in the grid affects the force acting on it. 
"""

def half_to_fullH(individual):                                                 # Half grid to full grid reflected in the horizontal
    lhs = individual
    rhs = flip(individual, 1)
    full_grid = concatenate((lhs, rhs), 1)
    return full_grid


def quarter_to_full(individual):                                               # Quarter grid to full grid
    ul = individual                                                            # Upper left
    ur = flip(individual, 1)                                                   # Upper right
    bl = flip(individual, 0)
    br = flip(bl, 1)

    top = concatenate((ul, ur), 1)
    bottom = concatenate((bl, br), 1)

    return concatenate((top, bottom), 0)


def full_to_fullSym(
    individual, side="left"
):  
    
    # Takes a normal grid and makes it symmetric by reflecting one of its halfs.

    if side == "left":
        lhalf = individual[:, : len(individual) // 2]
        rhalf = flip(lhalf, 1)

    elif side == "right":
        rhalf = individual[:, len(individual) // 2 :]
        lhalf = flip(rhalf, 1)

    else:
        raise ValueError("Side must be specified as either 'left' or 'right'")

    return concatenate((lhalf, rhalf), 1)
