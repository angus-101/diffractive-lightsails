"""
This program allows the user to view an optimised grid.
"""

from matplotlib import pyplot as plt
import numpy as np
from addaSeq_force_scoop import calculate_force_on_sample
from itertools import chain
from symmetric_adapters import half_to_fullH


def visualise_grid(arr):
    fig, ax = plt.subplots(figsize=(10, 10))                                   # Defining figure for the grid to appear on 
    ax.axis("off")
    ax.matshow(arr)
    plt.show()


def save_grid_visual(arr, name):                                               # Save the grid 
    plt.matshow(arr)
    plt.savefig(name)


def visualise_grid_file(fpath):

    with open(fpath, "rb") as f:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("off")
        ax.matshow(np.load(f))
        plt.show()


def visualise_sym_tile_file(fpath, tilefac):

    with open(fpath, "rb") as f:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("off")
        half_tile = np.load(f)
        full_tile = half_to_fullH(half_tile)
        grid = np.tile(full_tile, (tilefac, tilefac))
        ax.matshow(grid)
        plt.show()


def visualise_comparison(grids):

    size = int(np.ceil(np.sqrt(len(grids))))
    fig, ax = plt.subplots(size, size, figsize=(20, 20))
    for axis in chain.from_iterable(ax):
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    for ax, grid_item in zip(chain.from_iterable(ax), grids):
        grid, lam, title = grid_item
        ax.matshow(grid)
        force_val = calculate_force_on_sample(grid, scoop_=False, lam_frac_=1 / lam)[0]
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xlabel(f"Force: {force_val:.3g}", fontsize=10)

    plt.show()
    
# To show a grid, change the file name in the filepath to the name of the grid

if __name__ == "__main__":
    with open(
        r"C:\Users\angus\OneDrive - University of Bristol\University OneDrive\Documents\Year 4\Project\Coding\Data\grid50-0-25-03-2776523.npy",
        "rb",
    ) as grid_file:
        final_grid = np.load(grid_file)
        visualise_grid(final_grid)

