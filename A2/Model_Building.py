import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import time

# this is a model_building function, takes in 8 parameters, including number of conv_layer, layer_size, dense layer
# furthermore, training epoch, x and y for training and validation data
# this .py can not be ran as a script
def A2(conv_layer, layer_size, dense_layer, epoch, x_train, y_train, x_val, y_val):
    model = Sequential()

    model.add(   Conv2D(layer_size,(3,3),input_shape = x_train.shape[1:])   )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    for l in range(conv_layer-1):
        model.add(  Conv2D(layer_size,(3,3))  )
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    for l in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation("relu"))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss = "binary_crossentropy",
                 optimizer = "adam",
                 metrics = ["accuracy"])

    model.fit(x_train, y_train, batch_size=32, epochs=epoch, validation_data=(x_val, y_val))

    return model