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

Should be run with the plot_gen folder as the CWD
"""


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
        # Add first zero as there is no evaluation before first epoch
        serie = [0.0]
        with open(path + filename, "r") as logfile:
            for line in logfile:
                serie.append(float(line.strip()))
        data[model_name].append(serie)

    fig = plt.figure(dpi=400)
    # I comment these 3 lines when generating plot for fetchreach.
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

    for model_name, s_rates in data.items():
        success_rates = np.array(s_rates)
        sr_avgs = np.average(success_rates, 0)
        sr_errors = np.std(success_rates, 0)

        fill_up = sr_avgs + sr_errors
        fill_up[fill_up > 1] = 1
        fill_down = sr_avgs - sr_errors
        fill_down[fill_down < 0] = 0

        plt.plot(range(len(sr_avgs)), sr_avgs, label=model_name)
        plt.fill_between(range(len(sr_avgs)), fill_up, fill_down,
                         alpha=0.199, antialiased=True)

    plt.xlabel("Epochs")
    plt.ylabel("Success Rate")
    # I use the plt.legend() line when generating plot for fetchreach
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    plt.savefig('multi_output_plot.png', dpi=400)
    plt.close()


if __name__ == "__main__":
    generate_avg_plot()
