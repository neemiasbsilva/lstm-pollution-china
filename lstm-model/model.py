from keras.models import Sequential
from keras.layers import Dense, LSTM

def get_train_model(x_train):

    model = Sequential()

    model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))

    return model