import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import time

# potential number of dense_layer, layer sizes, number of conv_layer
dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

# pickle load all training and validation dataset
x_train = pickle.load(open(r"../Datasets/B2_dataset/x_train.pickle","rb"))
x_val = pickle.load(open(r"../Datasets/B2_dataset/x_val.pickle","rb"))
y_train = pickle.load(open(r"../Datasets/B2_dataset/y_train.pickle", "rb"))
y_val = pickle.load(open(r"../Datasets/B2_dataset/y_val.pickle", "rb"))

# this is for loop run all 27 combinations of layer choices
# each combination of layer choice is ran for 20 epochs
# train on training set and validate on validation set which are all load from B2_dataset folder pickle file
# 27 tensorboard logs are stored in selected folder, and can be viewed from tensorboard.exe
# therefore to choose most appropriate layer choice
# this file is already executed, simply type (tensorboard --logdir="log_path") in the cmd to view scalar plots

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir="..\Datasets\B2_dataset\logs\{}".format(NAME))

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=x_train.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(5))
            model.add(Activation("softmax"))

            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"])

            model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val), callbacks=[tensorboard])