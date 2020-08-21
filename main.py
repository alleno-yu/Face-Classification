# import
from A1.Pre_Processing import A1_data_preprocessing
from A1.Model_Building import A1
from A2.Pre_Processing import A2_data_preprocessing
from A2.Model_Building import A2
from B1.Pre_Processing import B1_data_preprocessing
from B1.Model_Building import B1
from B2.Pre_Processing import B2_data_preprocessing
from B2.Model_Building import B2
# ======================================================================================================================
# dataset path
A_img_path = r"Datasets/celeba/img"
A_label_path = r"Datasets/celeba/labels.csv"
B_img_path = r"Datasets/cartoon_set/img"
B_label_path = r"Datasets/cartoon_set/labels.csv"
# ======================================================================================================================
# Data preprocessing
A1_x_train, A1_x_test, A1_y_train, A1_y_test, A1_x_val, A1_y_val= A1_data_preprocessing(IMG_SIZE=70,
                                                                                        img_path=A_img_path,
                                                                                        label_path=A_label_path)
A2_x_train, A2_x_test, A2_y_train, A2_y_test, A2_x_val, A2_y_val = A2_data_preprocessing(IMG_SIZE=100,
                                                                                         img_path=A_img_path,
                                                                                         label_path=A_label_path)
B1_x_train, B1_x_test, B1_y_train, B1_y_test, B1_x_val, B1_y_val= B1_data_preprocessing(IMG_SIZE=70,
                                                                                        img_path=B_img_path,
                                                                                        label_path=B_label_path)
B2_x_train, B2_x_test, B2_y_train, B2_y_test, B2_x_val, B2_y_val= B2_data_preprocessing(IMG_SIZE=70,
                                                                                        img_path=B_img_path,
                                                                                        label_path=B_label_path)
# ======================================================================================================================
# Task A1
model_A1 = A1(conv_layer=3, layer_size=32, dense_layer=1,
              epoch=10, x_train=A1_x_train, y_train=A1_y_train, x_val=A1_x_val, y_val=A1_y_val)

_,acc_A1_train = model_A1.evaluate(A1_x_train,A1_y_train)
_,acc_A1_val = model_A1.evaluate(A1_x_val,A1_y_val)
_,acc_A1_test = model_A1.evaluate(A1_x_test,A1_y_test)
# ======================================================================================================================
# Task A2
model_A2 = A2(conv_layer=2, layer_size=64, dense_layer=0,
              epoch=6, x_train=A2_x_train, y_train=A2_y_train, x_val=A2_x_val, y_val=A2_y_val)

_,acc_A2_train = model_A2.evaluate(A2_x_train,A2_y_train)
_,acc_A2_val = model_A2.evaluate(A2_x_val,A2_y_val)
_,acc_A2_test = model_A2.evaluate(A2_x_test,A2_y_test)
# ======================================================================================================================
# Task B1
model_B1 = B1(conv_layer=1, layer_size=128, dense_layer=2,
           epoch=17, x_train=B1_x_train, y_train=B1_y_train, x_val=B1_x_val, y_val=B1_y_val)

_,acc_B1_train = model_B1.evaluate(B1_x_train,B1_y_train)
_,acc_B1_val = model_B1.evaluate(B1_x_val,B1_y_val)
_,acc_B1_test = model_B1.evaluate(B1_x_test,B1_y_test)
# ======================================================================================================================
# Task B2
model_B2 = B2(conv_layer=3, layer_size=64, dense_layer=2,
              epoch=15, x_train=B2_x_train, y_train=B2_y_train, x_val=B2_x_val, y_val=B2_y_val)

_,acc_B2_train = model_B2.evaluate(B2_x_train,B2_y_train)
_,acc_B2_val = model_B2.evaluate(B2_x_val,B2_y_val)
_,acc_B2_test = model_B2.evaluate(B2_x_test,B2_y_test)
# ======================================================================================================================
# Print out your results with following format:
print("""TA1_training_acc:{:.2f},    TA1_test_acc{:.2f},     TA1_val_acc{:.2f};
TA2_training_acc:{:.2f},    TA2_test_acc{:.2f},     TA2_val_acc{:.2f};
TB1_training_acc:{:.2f},    TB1_test_acc{:.2f},     TB1_val_acc{:.2f};
TB2_training_acc:{:.2f},    TB2_test_acc{:.2f},     TB2_val_acc{:.2f};""".format(acc_A1_train, acc_A1_test, acc_A1_val,
                                                                                 acc_A2_train, acc_A2_test, acc_A2_val,
                                                                                 acc_B1_train, acc_B1_test, acc_B1_val,
                                                                                 acc_B2_train, acc_B2_test, acc_B2_val))
