#coding=utf-8

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def SVM_train(Feature, Label,test_split):
    train_x, test_x, train_y, test_y = train_test_split(Feature, Label, test_size=1-test_split, random_state=42)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    test_x = min_max_scaler.transform(test_x)




    print("SVM training and testing...")
    SVM = SVC(kernel='rbf', C=3, gamma=2)
    SVM.fit(train_x, train_y)
    y_pred = SVM.predict(test_x)



    Acc = np.mean(y_pred == test_y)
    Conf_Mat = confusion_matrix(test_y, y_pred)
    Acc_N = Conf_Mat[0][0] / np.sum(Conf_Mat[0])
    Acc_V = Conf_Mat[1][1] / np.sum(Conf_Mat[1])
    Acc_R = Conf_Mat[2][2] / np.sum(Conf_Mat[2])
    Acc_L = Conf_Mat[3][3] / np.sum(Conf_Mat[3])

    return Acc, Acc_N, Acc_V, Acc_R, Acc_L, Conf_Mat
