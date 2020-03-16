import numpy as np
import pandas as pd
from model import get_train_model

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
