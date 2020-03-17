import numpy as np
import pandas as pd
from model import get_train_model
from sklearn.model_selection import train_test_split
import argparse
import os

parser = argparse.ArgumentParser(description="Using LSTM to predict the pollution of china")

parser.add_argument("-path_dataset", action="store", required=True, help="The dataset of the pollution china", dest="path_dataset")
parser.add_argument("-experiment_name", action="store", required=True, help="Folder to save the experiment", dest="experiment_name")

arguments = parser.parse_args()

path_dataset = arguments.path_dataset
experiment_name = arguments.experiment_name


# load the dataset
dataset = pd.read_csv("../pollution.csv", header=0, index_col=0)

# split into train and test
values = dataset.values

n_train_hours = 365 * 24

train = values[:n_train_hours,:]
test = values[n_train_hours:, :]

# split into input variables and output variables
x_train, y_train = train[:,:-1], train[:, -1]
x_test, x_test = test[:, :-1], test[:,-1]

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

model = get_train_model(x_train)

model.compile(loss='mse', optimizer='SGD')

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=121)

model.fit(x_train, y_train, epochs=100, batch_size=512, validation_data=(x_val, y_val), verbose=2, shuffle=False)

model.save(os.path.joint(experiment_name, "model.h5"))