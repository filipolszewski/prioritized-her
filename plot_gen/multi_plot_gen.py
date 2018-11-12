import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt

"""Utility script for generating plots with multiple data-series of 
success_rates for model comparisons.

Place all <model_name>_success_rate.log files in the success_rate_files 
directory and then run this script. Place multiple success_rate logs for each
 model to generate plot with std_dev intervals. 
 
Please specify model_name without using the '_' character as script is using 
it for the split method.

Should be run with the plot_gen folder as the CWD!"""


def generate_avg_plot():
    path = "success_rate_files/"
    filenames = [name for name in os.listdir(path)
                 if os.path.isfile(path + name)]
    if len(filenames) == 0:
        print("No files found in success_rate_files folder!")
        exit(1)

    data = defaultdict(list)
    for filename in filenames:
        model_name = filename.split('_')[0]
        serie = []
        with open(path + filename, "r") as logfile:
            for line in logfile:
                serie.append(float(line.strip()))
        data[model_name].append(serie)

    for model_name, s_rates in data.items():
        success_rates = np.array(s_rates)
        sr_avgs = np.average(success_rates, 0)
        sr_errors = np.std(success_rates, 0)

        plt.plot(range(len(sr_avgs)), sr_avgs, label=model_name)
        plt.fill_between(range(len(sr_avgs)), sr_avgs + sr_errors, sr_avgs -
                         sr_errors
                         , alpha=0.199, antialiased=True)
    plt.xlabel("Epochs")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig('multi_output_plot.png')
    plt.close()


if __name__ == "__main__":
    generate_avg_plot()
