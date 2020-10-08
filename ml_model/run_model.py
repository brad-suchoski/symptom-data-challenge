import h5py as h5
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import csv

from datetime import datetime, date
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#--------------------------------
# Set all modifiable variables for the script 
num_in, num_out, offset = 21, 21, 5
dropout = 0.2
nodes = 64
num_datasets = 100 
epochs = 5
batch_size = 64
val_split = 0.2
num_epoch_sets = 6
inputs_name = "/dataset.h5"
results_name = "/results.h5"
#--------------------------------

def setup_model(input_shape):
    model = keras.Sequential()
    model.add(layers.LSTM(nodes, input_shape=(num_in, input_shape),
                      dropout = dropout, return_sequences=True))
    model.add(layers.LSTM(nodes, dropout = dropout, return_sequences=True))
    model.add(layers.LSTM(nodes, dropout = dropout, return_sequences=True))
    model.add(layers.LSTM(nodes, dropout = dropout, return_sequences=True))
    model.add(layers.LSTM(nodes, dropout = dropout, return_sequences=False))
    model.add(layers.Dense(nodes/2, activation = 'elu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_out, activation = 'elu'))
    model.compile(loss='mse', optimizer='adam')
    return model

#--------------------------------
averages = np.ndarray((0,), dtype=np.float32)
medians = np.ndarray((0,), dtype=np.float32)

with h5.File(inputs_name, "r") as fin, h5.File(results_name, "w") as fout:
    for i in range(0, num_datasets):
        print("Dataset: " + str(i))
        this_err_ave = np.ndarray((0,), dtype=np.float32)
        this_err_med = np.ndarray((0,), dtype=np.float32)

        # Read in the dataset from the h5 file
        group_name = "group" + str(i)
        file_tr_dataset = group_name + "/train_dataset"
        file_tr_labels = group_name + "/train_labels"
        file_te_dataset = group_name + "/test_dataset"
        file_te_labels = group_name + "/test_labels"
        file_scale = group_name + "/label_scale"
        
        train_dataset = fin[file_tr_dataset][()]
        train_labels  = fin[file_tr_labels][()]
        test_dataset = fin[file_te_dataset][()]
        test_labels = fin[file_te_labels][()]
        label_scale = fin[file_scale][()]
        
        this_preds = np.ndarray((0, test_dataset.shape[0], num_out), dtype=np.float32)

        model = setup_model(train_dataset.shape[2])
        # Run all of the epoch sets, storing the results
        for j in range(0, num_epoch_sets):
            history = model.fit(train_dataset, train_labels, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=0)
            pred = model.predict(test_dataset)
            this_preds = np.append(this_preds, pred.reshape(1,pred.shape[0],pred.shape[1]), axis=0)
            ave_error = np.absolute((test_labels[:,6]-pred[:,6]) / test_labels[:,6] * 100)
            this_err_ave = np.append(this_err_ave, np.average(list(filter(np.isfinite, ave_error))))
            this_err_med = np.append(this_err_med, np.median(ave_error))
        # END FOR ALL EPOCHS
        
        best = np.argmin(this_err_ave)  # Ranks best by average
        best_err_ave = this_err_ave[best]
        best_err_med = this_err_med[best]
        averages = np.append(averages, best_err_ave)
        medians = np.append(medians, best_err_med)
        best_pred = this_preds[best] * label_scale

        # Write the best epoch groups results to the output file
        fout.create_group(group_name)
        file_best = group_name + "/best"
        file_err_ave = group_name + "/err_ave"
        file_err_med = group_name + "/err_med"
        file_pred = group_name + "/pred"
        fout[file_best] = best
        fout[file_err_ave] = best_err_ave
        fout[file_err_med] = best_err_med
        fout[file_pred] = best_pred
    # END FOR ALL DATASETS
    # END WRITE TO FILE
    # END FOR EACH 0 - num_datasets
# END READ INPUT FILE
