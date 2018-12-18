#coding=utf-8
from CNN import *
from LSTM import *
from SVM import *
import numpy as np
import h5py as hp
def load_data(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    print(data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
        print (k,v)
    print (arrays_d)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

if __name__ == "__main__":

    path_data = "./data/DL_data.mat"
    origin_data = load_data(path_data,"Data",dtype='float32')
    path_data = "./data/DL_label.mat"
    label = load_data(path_data,"Label",dtype='float32')
    test_split = 0.7
    print("CNN_train:")
    Acc, Acc_N, Acc_V, Acc_R, Acc_L, Conf_Mat =CNN_train(origin_data, label, test_split)
    print ("\nAcc:")
    print (Acc)
    print("\nAcc_N:")
    print(Acc_N)
    print("\nAcc_V:")
    print(Acc_V)
    print ("\nAcc_R:")
    print(Acc_R)
    print("\nAcc_L:")
    print(Acc_L)
    print("\nConf_Mat:")
    print(Conf_Mat)
    '''
    print("LSTM_training:")
    Acc, Acc_N, Acc_V, Acc_R, Acc_L, Conf_Mat = LSTM_train(origin_data, label, test_split)
    print("\nAcc:")
    print(Acc)
    print("\nAcc_N:")
    print(Acc_N)
    print("\nAcc_V:")
    print(Acc_V)
    print("\nAcc_R:")
    print(Acc_R)
    print("\nAcc_L:")
    print(Acc_L)
    print("\nConf_Mat:")
    print(Conf_Mat)

    test_split = 0.5
    FeatureFile = './data/ML_data.mat'
    LabelFile = './data/ML_label.mat'


    Feature = load_data(FeatureFile, 'Feature')
    Feature = Feature.T
    Label = load_data(LabelFile, 'Label')
    Label = Label.squeeze()
    Acc, Acc_N, Acc_V, Acc_R, Acc_L, Conf_Mat = SVM_train(Feature, Label, test_split)
    print ("SVM training\n")
    print("\nAcc:")
    print(Acc)
    print("\nAcc_N:")
    print(Acc_N)
    print("\nAcc_V:")
    print(Acc_V)
    print("\nAcc_R:")
    print(Acc_R)
    print("\nAcc_L:")
    print(Acc_L)
    print("\nConf_Mat:")
    print(Conf_Mat)'''

