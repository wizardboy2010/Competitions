from keras.models import Sequential
from keras.layers import Dense, Activation

INPUT_SIZE = 30

def model_1(input_size):
    model = Sequential()
    model.add(Dense(32, input_dim=input_size, Activation = 'relu'))
    model.add(Dense(1, Activation = 'sigmoid'))

    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
