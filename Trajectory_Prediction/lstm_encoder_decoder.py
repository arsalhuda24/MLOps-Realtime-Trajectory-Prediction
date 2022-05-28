import tensorflow as tf
from keras.layers import LSTM, Dense, RepeatVector, Dropout, TimeDistributed
from keras import optimizers
from keras.models import Sequential
# from sgan.data.trajectories import TrajectoryDataset
import matplotlib.pyplot as plt
from keras.models import load_model
import torch
import os
# from sgan.losses import displacement_error, final_displacement_error
from keras.layers import embeddings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir("/home/asyed/my_docker/Trajectory_prediction/sgan")
# from sgan.data.trajectories import TrajectoryDataset, seq_collate
from preprocess import TrajectoryDataset

from keras import optimizers
from tensorflow import keras



hidden_neurons= 128
tf.config.run_functions_eagerly(True)


model= Sequential()
model.add(Dense(64, input_shape=(None,2),activation="linear"))
model.add(LSTM(128, input_shape=(8,64), return_sequences= False))
model.add(RepeatVector(12))
model.add(LSTM(128, return_sequences= True))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(2,activation= "linear")))
#optim= optimizers.adam_v2(lr=0.001)
optim = keras.optimizers.Adam(learning_rate=0.001)


model.compile(loss="mse", optimizer=optim, metrics=["accuracy"])


