import h5py as h5
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import csv
from scipy import stats

from datetime import datetime, date
import matplotlib.pyplot as plt

#--------------------------------
# Set all modifiable variables for the script 
num_in, num_out, offset = 21, 21, 5
num_datasets = 100
inputs_name = "dataset.h5"
results_name = "results.h5"
#--------------------------------

with h5.File(inputs_name, "r") as fin, h5.File(results_name, "r") as fres:
    averages = np.ndarray((0,), dtype=np.float32)
    medians = np.ndarray((0,), dtype=np.float32)
    surges = np.ndarray((0,), dtype=np.float32)
    surge_ave = np.ndarray((0,), dtype=np.float32)
    surge_med = np.ndarray((0,), dtype=np.float32)
    nonsurge_ave = np.ndarray((0,), dtype=np.float32)
    nonsurge_med = np.ndarray((0,), dtype=np.float32)

    for i in range(0, num_datasets):
        this_err_ave = np.ndarray((0,), dtype=np.float32)
        this_err_med = np.ndarray((0,), dtype=np.float32)
        this_preds = np.ndarray((0,num_out), dtype=np.float32)
        
        # Import the data from the original input file
        group_name = "group" + str(i)
        file_tr_dataset = group_name + "/train_dataset"
        file_tr_labels = group_name + "/train_labels"
        file_te_dataset = group_name + "/test_dataset"
        file_te_labels = group_name + "/test_labels"
        file_scale = group_name + "/label_scale"
        file_surge = group_name + "/test_surge"
        
        train_dataset = fin[file_tr_dataset][()]
        train_labels  = fin[file_tr_labels][()]
        test_dataset = fin[file_te_dataset][()]
        test_labels = fin[file_te_labels][()]
        label_scale = fin[file_scale][()]
        test_surge = fin[file_surge][()]

        surges = np.nonzero(test_surge)[0]
        nonsurges = np.where(test_surge == 0)[0]

        # Import data from the results output file
        file_preds = group_name + "/pred"
        pred = fres[file_preds][()]
        test_labels = test_labels * label_scale

        error = (test_labels - pred) / test_labels * 100 
        averages = np.append(averages, np.average(error))
        medians = np.append(medians, np.median(error))

        # Split datasets into surge and nonsurge groups
        if(len(surges) > 0):
            surge_preds = pred[surges]
            surge_ave_err = ((test_labels[surges]-pred[surges]) / test_labels[surges]) * 100
            surge_ave = np.append(surge_ave, np.average(surge_ave_err))
            surge_med = np.append(surge_med, np.median(surge_ave_err))
        nonsurge_preds = pred[nonsurges]
        nonsurge_ave_err = ((test_labels[nonsurges]-pred[nonsurges]) / test_labels[nonsurges]) * 100
        nonsurge_ave = np.append(nonsurge_ave, np.average(nonsurge_ave_err))
        nonsurge_med = np.append(nonsurge_med, np.median(nonsurge_ave_err))
    # END FOR EACH 0 - num_datasets

    # Calculate and print the statistics
    overall_average = np.average(averages)
    average_median = np.average(medians)
    print("overall_average: ", overall_average, ", average_median: ", average_median)
    print("Averages Description:")
    print(stats.describe(averages))
    print("25th: ", np.percentile(averages, 25), ", 75th: ", np.percentile(averages, 75))
    print("Medians Description:")
    print(stats.describe(medians))
    print("25th: ", np.percentile(medians, 25), ", 75th: ", np.percentile(medians, 75))
    print("----------------------------------------------")
    if(len(surges) > 0):
        surge_average = np.average(surge_ave)
        surge_median = np.average(surge_med)
        print("surge overall_average: ", surge_average, ", average_median: ", surge_median)
        print("Averages Description:")
        print(stats.describe(surge_ave))
        print("25th: ", np.percentile(surge_ave, 25), ", 75th: ", np.percentile(surge_ave, 75))
        print("Medians Description:")
        print(stats.describe(surge_med))
        print("25th: ", np.percentile(surge_med, 25), ", 75th: ", np.percentile(surge_med, 75))
    print("----------------------------------------------")
    nonsurge_average = np.average(nonsurge_ave)
    nonsurge_median = np.average(nonsurge_med)
    print("nonsurge overall_average: ", nonsurge_average, ", average_median: ", nonsurge_median)
    print("Averages Description:")
    print(stats.describe(nonsurge_ave))
    print("25th: ", np.percentile(nonsurge_ave, 25), ", 75th: ", np.percentile(nonsurge_ave, 75))
    print("Medians Description:")
    print(stats.describe(nonsurge_med))
    print("25th: ", np.percentile(nonsurge_med, 25), ", 75th: ", np.percentile(nonsurge_med, 75))

# END READ INPUT FILE
