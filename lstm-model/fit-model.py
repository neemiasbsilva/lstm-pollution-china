import numpy as np
import pandas as pd

# load the dataset
dataset = pd.read_csv("../pollution.csv", header=0, index_col=0)

# split into train and test
values = dataset.values

n_train_hours = 365 * 24

train = values[:n_train_hours,:]
test = values[n_train_hours:, :]

