import numpy as np
import os
import matplotlib.pyplot as plt

"""Utility script for generating averaged plots with std_dev intervals, used
to plot averaged success_rates for multiple runs of the same agent. 

Place all success_rate.log files in the success_rate_files directory and then
run this script. 

Should be run with the plot_gen folder as the CWD!"""


def generate_avg_plot():
    path = "success_rate_files/"
    num_files = len([name for name in os.listdir(path)
                     if os.path.isfile(path + name)])
    if num_files == 0:
        print("No files found in success_rate_files folder!")
        exit(1)

    data = []
    for i in range(num_files):
        serie = []
        with open((path + "{}.log").format(i + 1), "r") as logfile:
            for line in logfile:
                serie.append(float(line.strip()))
        data.append(serie)

    success_rates = np.array(data)
    sr_avgs = np.average(success_rates, 0)
    sr_error = np.std(success_rates, 0)

    plt.plot(range(len(sr_avgs)), sr_avgs)
    plt.fill_between(range(len(sr_avgs)), sr_avgs + sr_error, sr_avgs - sr_error
                     , alpha=0.199, antialiased=True)
    plt.savefig('output_plot.png')
    plt.close()


if __name__ == "__main__":
    generate_avg_plot()