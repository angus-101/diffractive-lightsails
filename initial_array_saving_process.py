#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:19:48 2021

@author: angusmcconnell
"""

"""
This program saves the array grid for use with ADDA and DEAP. It starts off
as a random array of ones and zeros, and DEAP optimises it to maximise the 
force calulcated by ADDA.
"""

import numpy as np 

N = 100
grid = np.random.choice([0, 1], size=[N,N], p=[.5, .5])
array_grid_4 = np.asarray(grid)
print(array_grid_4, len(array_grid_4))
np.save('array_grid_4.npy', array_grid_4)