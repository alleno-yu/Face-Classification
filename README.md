#File Organization
##1. File Structure
#####1.1 this project folder contains 5 sub-folders, including A1, A2, B1, B2 and Datasets
#####1.2 A1, A2, B1, B2 each contains code of corresponding task. There are three files, Pre_Processsing.py, Model_Training.py, Model_Building.py.
#####1.3 Datasets should contain 6 folders, they are 4 folders including A1_dataset, A2_dataset, B1_dataset, B2_dataset and 2 other training dataset folder, celeb and cartoon-set.
#####1.4 each Task_dataset folder contains logs file of tensorboard and pickle files of training array and validation array.
#####1.5 There are 5000 images and a csv file in celeb folder, 10000 images and a csv file in cartoon-set folder. But these two folders will be empty when submitting project.
##2.File Contents
#####2.1 Pre_Processing.py is used to pre-process the images, including convert images into array, normalizing array, resizing, spliting dataset and etc.
#####2.2 Also, Pre_Processing.py can be run as a script or import as a function in the main.py. If we run this .py as a script, it will generate 4 pickle files contain X_train, Y_train, X_val and Y_val.
#####2.3 Model_Training.py can only be run as a script, it has no contribion to the main.py, but this file is extremely useful.
#####2.4 Model_Training.py tests over a number of potential layer sets, and generate log files using tensorboard. We can view the plotted graph of accuracy and loss for each epoch by using tensorboard.exe.
#####2.5 Model_Building.py can only be import as a function in the main.py, this helps us to build wanted model and test from there.
##3. Package dependencies
#####3.1 Python3
#####3.2 opencv 4.1.2, numpy 1.17.4, matplotlib 2.2.2, os, tensorflow-gpu 2.0, pickle, time libraries.
##4. Tensorboard usage
#####4.1 Tensorboard is used to analyse and make best choice of model, it can plot the accuracy or loss against each epoch for all possible models.
#####4.2 To use tensorboard, first install tensorbaord using anaconda.
#####4.2 Open anaconda cmd window, type the following code
##### _tensorboard --logdir="log_file_path"_
#####4.3 The logs file contains the plot of accuracy and loss of 27 models for each task.
#####4.4 To choose the best model for each task, choose the one with the lowest and the most stable validation loss
#####4.5 Once model is chosen, dense_layer, conv_layer, layer_size, epoch parameters are prepared for the main.py for accuracy test 
##5. Comments
#####5.1 since the structure of code for each task is similar, repetitive comments would only be made under TA1.
#####5.2 summary comments of function are still available in each file




