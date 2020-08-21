import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

# this is data preprocessing function, takes three arguments(img size, img path and csv file path)
# it runs over all imgs under img_path root and convert into array using cv2.imread function
# it this function returns 6 datasets arrays, x and y for training, val and test.
# the data set is split into 70% for training, 20% for validation and 10% for testing
# the array data is already normalized, results are between 0 and 1
def A1_data_preprocessing(IMG_SIZE, img_path, label_path):
    # create an empty array, training data
    training_data = []

    # open label file by define label path
    labels_file = open(label_path, 'r')
    # use readlines function to read csv file
    lines = labels_file.readlines()
    # extract facial labels, and build a dictionary called facial label.
    # key is the index of images, value is the facial label
    facial_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}

    for img in os.listdir(img_path):  # for each image in the path, img = 1.png for example
        # convert facial label from (-1,1) to (0, 1)
        class_num = int((facial_labels[img.split(".")[0]]+1)/2)
        try:   # prevent corrupted image interrupt the data-processing
            # convert images into gray scale, then into array
            img_array = cv2.imread(os.path.join(img_path, img), cv2.IMREAD_GRAYSCALE)
            # resize the image_array into wanted size, in this case is 70x70
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            # append training data with new_array and class_num
            training_data.append([new_array, class_num])
        except Exception as e:
            pass

    # create X and y as empty array
    X = []
    y = []

    # first element in each row is features, and second is label
    for features, label in training_data:
        # append X and y with feature and labels
        X.append(features)
        y.append(label)

    # reshape the X for future processing
    # in this case, X is reshaped into (5000, 70, 70, 1)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    # normalize X between 0 and 1, so we can get better training result
    X = tf.keras.utils.normalize(X, axis=1)

    # split training data, validation data, test data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.222222, random_state=0)


    return x_train, x_test, y_train, y_test, x_val, y_val

if __name__ == "__main__":
    # path for img folder and csv file
    # these paths are relative path, pls put celeba folder under Datasets folder
    img_path = r"../Datasets/celeba/img"
    label_path = r"../Datasets/celeba/labels.csv"

    #call data processing functions, imgs are resized into 70x70
    A1_x_train, A1_x_test, A1_y_train, A1_y_test, A1_x_val, A1_y_val = A1_data_preprocessing(IMG_SIZE=70,
                                                                                             img_path=img_path,
                                                                                             label_path=label_path)
    # use pickle to store splited training dataset and validation dataset
    # this file is either called from main, or run as script to produce pickle file for model training use
    print(A1_x_train.shape)
    pickle_out = open("../Datasets/A1_dataset/x_train.pickle", "wb")
    pickle.dump(A1_x_train, pickle_out)
    pickle_out.close()

    print(A1_x_val.shape)
    pickle_out = open("../Datasets/A1_dataset/x_val.pickle", "wb")
    pickle.dump(A1_x_val, pickle_out)
    pickle_out.close()

    print(A1_y_train.shape)
    pickle_out = open("../Datasets/A1_dataset/y_train.pickle", "wb")
    pickle.dump(A1_y_train, pickle_out)
    pickle_out.close()

    print(A1_y_val.shape)
    pickle_out = open("../Datasets/A1_dataset/y_val.pickle", "wb")
    pickle.dump(A1_y_val, pickle_out)
    pickle_out.close()

    print(A1_x_test.shape)
    pickle_out = open("../Datasets/A1_dataset/x_test.pickle", "wb")
    pickle.dump(A1_x_test, pickle_out)
    pickle_out.close()

    print(A1_y_test.shape)
    pickle_out = open("../Datasets/A1_dataset/y_test.pickle", "wb")
    pickle.dump(A1_y_test, pickle_out)
    pickle_out.close()