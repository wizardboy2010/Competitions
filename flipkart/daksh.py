import pandas as pd
#import gspread
#from gspread_dataframe import get_as_dataframe, set_with_dataframe
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras import regularizers, optimizers
import os
import zipfile
import io
from PIL import Image
import tqdm
from keras_preprocessing.image import ImageDataGenerator

train =pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=train, directory="data/edge_data",
                                            x_col="image_name", y_col=["x1","x2","y1","y2"], class_mode= "other",
                                            target_size=(640,480), batch_size=32, shuffle=False)
y_train = train.drop("image_name", axis=1)

model = Sequential()

model.add(Conv2D(32,(4,4),padding = 'same',input_shape=(640,480,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (4, 4)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))

model.compile(optimizers.rmsprop(lr=0.0001),
loss="mean_squared_error", metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

train_count = 0
def generator(data, n):
    global train_count
    if train_count >= train_generator.n:
        train_count = 0
    while True:
        batch_labels = data[train_count:train_count+n]
        train_count += n
        print(train_count)
        return np.array(batch_labels)

def combined_generator(gen1, y_data):
    x = gen1.next()
    yield x, generator(y_data, len(x))

model.fit_generator(generator=train_generator,#combined_generator(train_generator, y_train),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=2)