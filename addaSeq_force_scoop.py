#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:52:45 2021

@author: angusmcconnell
"""

"""
This program runs the discrete dipole approximation (via ADDA) on a numpy 
array filled with ones and zeros, representing material and no material,
respectively. The force acting on the array is calculated. This result can
be passed to the DEAP program to act as a fitness of an individual.
"""

import subprocess
import os
import numpy as np
from time import process_time
from random import uniform
import math
import scoop

# https://github.com/adda-team/adda/blob/master/src/CalculateE.c (617) to make fileIO redundant if wanted tor rewrite

class AddaException(Exception):
    pass


def run_adda_force(dipole_per_lambda, shape_file, output_dir_name, working_directory, wavelength, real_ref_index, im_ref_index): #used in function below 'calculate_force_on_sample'
    
    """
    Runs the ADDA program, using a subprocess, formatted with the correct
    parameters and then returns the path to the results file.

    Args:
        dipole_per_lambda (int): Dipoles per lambda parameter 
        shape_file (string): Path to the file which describes the shape in terms of its dipole co-ordinates
        output_dir_name (string): Name of the folder for Adda to store results.
        working_directory (string): Directory path for adda to work in/store temporary results
        wavelength (float): wavelength of incoming radiation in micrometers

    Raises:
        AddaException: Custom exception raised if there is a problem encountered running Adda

    Returns:
        string: Path to the folder containing the simulation results
    """

    pwd = os.getcwd()                                                          # Previous working directory
    os.chdir(working_directory)
    wd = os.getcwd()                                                           # Current working directory where ADDA is 

    process = subprocess.Popen(                                                # Passes arguments as a sequence to be used in the ADDA program
        [
            "adda",                                                            # ADDA program name    
            "-Cpr",                                                            # Outputs a force measurement to be read by function "read_force"
            "-lambda",    
            str(wavelength),                                                   # Measured in micrometers
            "-dpl",
            str(dipole_per_lambda),
            "-m",
            str(real_ref_index),
            str(im_ref_index),
            "-shape",
            "read",
            shape_file,                                                        # The binary input shape file
            "-dir",
            output_dir_name,                                                   # Where info is stored
        ],
        stdout=subprocess.PIPE,                                                # Ensures that the output is given to the mother process(here)
        stderr=subprocess.PIPE,                                                # Passes the error to the mother function (ie from ADDA to this program)
        )
    _, stderr = process.communicate()                                          # Communicates stderr information to python if there exists an error in execution

    os.chdir(pwd)

    if stderr:
        raise AddaException(stderr.decode("utf-8"))                            # Decodes error into utf-8 format

    return wd + '/' + output_dir_name


def gen_shape_file(shape_arr, path_to_wd, identifier):
    
    """
    Takes a grid of booleans representing the shape of the dipoles
    and converts it to a file that can be interpreted by ADDA

    Args:
        shape_arr (numpy 2D array): 2D Grid representing shape geometry 
        path_to_wd (string): Path to working directory, where ADDA is working from
        identifier (string): Unique number/code to give shape file 

    Returns:
        string: Path to newly created shape file
    """
    
    file_path = f"{path_to_wd}/shape{identifier}.txt"

    with open(file_path, "w") as shape_file:                                   # Opens the shape boolean file to write 
        for ix, iy in np.ndindex(shape_arr.shape):                             # Returns list of all possible index values of the shape_arr
            if shape_arr[ix, iy]:
                for k in range(4):                                             # Add entries for all three components x,y,z, where z goes up to the number of dipoles in the range
                    print(ix, iy, k, file=shape_file)                          # Prints the x y z coordinates of the dipoles in the shape_file 

    return file_path


def read_force(results_dir):
    
    """ 
    Reads force values from ADDA output files and returnsas float value for later use.

    Args:
        results_dir (string): Path to directory where ADDA process has stored results

    Returns:
        float: component magnitude
    """

    with open(
        f"{results_dir}/CrossSec-X", "r"
    ) as force_px_file:                                                        # Additions due to X polarisation of light
        [next(force_px_file) for i in range(8)]                                # Skip to relevant line
        raw = next(force_px_file)
        fpx_x, fpx_y, fpx_z = raw.split("=")[1].strip()[1:-1].split(",")
        fpx_x, fpx_y, fpx_z = float(fpx_x), float(fpx_y), float(fpx_z)

    with open(
        f"{results_dir}/CrossSec-Y", "r"
    ) as force_py_file:                                                        # Additions due to Y polarisation of light
        [next(force_py_file) for i in range(8)]
        raw = next(force_py_file)
        fpy_x, fpy_y, fpy_z = raw.split("=")[1].strip()[1:-1].split(",")
        fpy_x, fpy_y, fpy_z = float(fpy_x), float(fpy_y), float(fpy_z)
        
    F_x = (fpx_x + fpy_x)/8*math.pi                                            # Equation 66 in ADDA manual states that F=C_pr/8*pi (assuming normalised E-field and in a vacuum)
    F_y = (fpx_y + fpy_y)/8*math.pi
    F_z = (fpx_z + fpy_z)/8*math.pi 

    return math.sqrt(F_x**2 + F_y**2 + F_z**2)        


def calculate_force_on_sample(
    shape_arr,
    lam_frac_,
    working_directory_= r"C:\Users\angus\OneDrive - University of Bristol\University OneDrive\Documents\Year 4\Project\Coding\ADDA\adda-1.4.0_Compiled\win64", 
    del_files_= True,
    scoop_= False,
    ):
    
    """
    Calculates force on a given numpy 2d grid of dipoles representing a given shape

    Args:
        shape_arr (numpy 2d array): Grid of dipoles representing shape read from external file
        working_directory_ (str, optional): Where ADDA should run with temporary files.
        del_files_ (bool, optional): Flag for adda to remove files after running. Defaults to True.
        scoop_ (bool, optional): Flag to enable scoop multiprocessing. Defaults to True.

    Returns:
        float : Radiation force produced
    """

    # Input parameters
    
    wavelength = 350                                                           # In micrometers
    real_ref_index = 5                                                         # Real part of refractive index 
    im_ref_index = 3                                                           # Imaginary part of refractive index
    dipole_per_lambda = lam_frac_ * len(
        shape_arr
        )                                                                      # Fixes grid to be 1/lam_frac_ wavelengths wide
    experiment_identifier = 1 if not scoop_ else scoop.worker.decode("utf-8")
    experiment_identifier = experiment_identifier[-1]
    shape_path = gen_shape_file(shape_arr, working_directory_, experiment_identifier) # Path to dipole shape storage

    result_path = run_adda_force(
        dipole_per_lambda,
        f"shape{experiment_identifier}.txt",
        f"experiment{experiment_identifier}",
        working_directory_,
        wavelength,
        real_ref_index,
        im_ref_index
    )

    force = read_force(result_path)

    # Cleanup to stop file system getting clogged, deletes all files where the ADDA data was stored during calculation
    
    if del_files_:
        os.system(f"rm {shape_path}")
        os.system(f"rm -r {result_path}")        
    return (force,)                                                            # Returns a single value array to be compatible with Deap framework

# Calculate force on a shape 
      
if __name__ == "__main__":
    with open(
        r"C:\Users\angus\OneDrive - University of Bristol\University OneDrive\Documents\Year 4\Project\Coding\array_grid_4.npy",  
        "rb",
    ) as grid_file:
        components = calculate_force_on_sample(np.load(grid_file))
        print(f"component of force: {components}")