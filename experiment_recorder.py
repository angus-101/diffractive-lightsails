"""
This program saves the results of the evolutionary algorithm and saves them
to a spreadsheet.
"""

import pandas as pd
from datetime import datetime
from time import time
import numpy as np
import pickle
from matplotlib import pyplot as plt
from random import randint


class ExpRecord:
    
    # Utility class for recording data about optimisation runs, useful for comparing methods.

    def __init__(self, path_to_records):
        self.directory_path = path_to_records + "/"
        self.data_frame = pd.read_csv(
            self.directory_path + "ExperimentData.csv", parse_dates=["Date"]
        )

    def save(self):                                                            # Saves any new runs to the file
        self.data_frame.to_csv(self.directory_path + "ExperimentData.csv", index=False)

    def add_experiment(
        self,
        grid,
        log,
        force,
        direc,
        n_gen,
        pop,
        cx_p,
        cx_meth,
        cx_param,
        mut_p,
        mut_meth,
        mut_param,
        sel_meth,
        sel_param,
    ):                                                                         # Add details to the table from an optimisation run
        grid_size = len(grid)
        red_date = datetime.today().strftime("%d-%m")
        time_now = time()
        ten_sec = int(time_now % 100)
        sec = int(time_now % 10)
        tenth = int(time_now % 1 * 10)
        hund = int(time_now % 0.1 * 100)
        r_num1 = randint(0, 10)                                                # Random numbers to ensure no file name clashes
        r_num2 = randint(0, 10)

        grid_file_name = f"grid{grid_size}-{direc}-{red_date}-{ten_sec}{sec}{tenth}{hund}{r_num1}{r_num2}.npy"
        log_file_name = f"log{grid_size}-{direc}-{red_date}-{ten_sec}{sec}{tenth}{hund}{r_num1}{r_num2}.pkl"

        with open(self.directory_path + grid_file_name, "wb") as f:
            np.save(f, grid)

        with open(self.directory_path + log_file_name, "wb") as f:
            pickle.dump(log, f)

        row_to_add = {
            "Date": pd.to_datetime("today"),
            "Grid Size": grid_size,
            "Force": force,
            "Direction": direc,
            "Number of Generations": n_gen,
            "Population": pop,
            "Cross Over Prob": cx_p,
            "Cross Over Method": cx_meth,
            "Cross Over Parameter": cx_param,
            "Mutation Probability": mut_p,
            "Mutation Method": mut_meth,
            "Mutation Parameter": mut_param,
            "Selection Method": sel_meth,
            "Selection Parameter": sel_param,
            "Grid File Name": grid_file_name,
            "Log File name": log_file_name,
        }

        self.data_frame = self.data_frame.append(row_to_add, ignore_index=True)

    def __repr__(self):
        return repr(self.data_frame)

