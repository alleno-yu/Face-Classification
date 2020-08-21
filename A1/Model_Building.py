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
def A1(conv_layer, layer_size, dense_layer, epoch, x_train, y_train, x_val, y_val):
    # sequential model
    model = Sequential()

    # add a first layer---Conv2D layer
    # filters is 32
    # filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    # kernel size is 3x3
    # When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    # e.g. `input_shape=(70, 70, 1)` for 70x70 gray pictures
    # Input shape: channel_last gives a 4D tensor with shape:(batch, rows, cols, channels)---(?, 70, 70, 1)
    # Output shape of conv2D: channel_last gives a 4D tensor with shape:(batch, new_rows, new_cols, filters)---
    # (?, 68, 68, 32)
    model.add(   Conv2D(layer_size,(3,3),input_shape = x_train.shape[1:])   )
    # Output shape of activation relu: (?, 68, 68, 32)
    model.add(Activation("relu"))
    # Output shape: (batch_size, pooled_rows, pooled_cols, channels)
    # Output shape of MaxPooling2D: (?, 34, 34, 32)
    # In every 2x2 pool, only keep one. This means, the dimensionality of rows and cols is halved
    model.add(MaxPooling2D(pool_size=(2,2)))

    # decide whether to add more conv layer, but without input shape
    # if one more conv_layer is used, input shape: (?, 34, 34, 32)
    for l in range(conv_layer-1):
        # output shape of conv2D is (?, 32, 32, 32)
        model.add(  Conv2D(layer_size,(3,3))  )
        # output shape of relu funtion is (?, 32, 32, 32)
        model.add(Activation("relu"))
        # output shape os maxpooling2D is (?, 16, 16, 32)
        model.add(MaxPooling2D(pool_size=(2,2)))

    # channels are getting bigger, 1D array, output shape of flatten layer: (?, 8192)
    # but do not affect batch size
    model.add(Flatten())

    # potential dense_layer
    for l in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation("relu"))

    # dense 8192 nodes into 1
    # output shape of dense layer: (?, 1)
    model.add(Dense(1))
    # activation layer of sigmoid
    model.add(Activation("sigmoid"))

    # loss function, optimizer and metrics choices
    model.compile(loss = "binary_crossentropy",
                 optimizer = "adam",
                 metrics = ["accuracy"])

    # fit model with x_train and y_train, batch size=32, validate with x_val and y_val
    model.fit(x_train, y_train, batch_size=32, epochs=epoch, validation_data=(x_val, y_val))

    return model
# if __name__ == "__main__":
#     A1_x_train = pickle.load(open(r"../Datasets/A1_dataset/x_train.pickle","rb"))
#     A1_x_val = pickle.load(open(r"../Datasets/A1_dataset/x_val.pickle","rb"))
#     A1_y_train = pickle.load(open(r"../Datasets/A1_dataset/y_train.pickle", "rb"))
#     A1_y_val = pickle.load(open(r"../Datasets/A1_dataset/y_val.pickle", "rb"))
#     A1_x_test = pickle.load(open(r"../Datasets/A1_dataset/x_test.pickle", "rb"))
#     A1_y_test = pickle.load(open(r"../Datasets/A1_dataset/y_test.pickle", "rb"))
#
#     model_A1 = A1(conv_layer=3, layer_size=32, dense_layer=1,
#                   epoch=10, x_train=A1_x_train, y_train=A1_y_train, x_val=A1_x_val, y_val=A1_y_val)
#     _,acc_A1_train = model_A1.evaluate(A1_x_train,A1_y_train)
#     _,acc_A1_val = model_A1.evaluate(A1_x_val,A1_y_val)
#     _, acc_A1_test = model_A1.evaluate(A1_x_test, A1_y_test)
#     print(acc_A1_train, acc_A1_val,  acc_A1_test)
