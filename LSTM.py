#coding=utf-8
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess=tf.InteractiveSession()


def LSTM_network(x, n_input, n_hidden, n_steps, n_classes):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    for i in range(n_steps):
        if i == 0:
            fv = outputs[0]
        else:
            fv = tf.concat([fv, outputs[i]], 1)
    fvp = tf.reshape(fv, [-1, 1, n_steps * n_hidden, 1])
    shp = fvp.get_shape()
    flatten_shape = shp[1].value * shp[2].value * shp[3].value

    fvp2 = tf.reshape(fvp, [-1, flatten_shape])

    weights = tf.Variable(tf.random_normal([flatten_shape, n_classes]))
    biases = tf.Variable(tf.random_normal([n_classes]))

    return tf.matmul(fvp2, weights) + biases


def get_batch(train_x,train_y,batch_size):
    indices=np.random.choice(train_x.shape[0],batch_size,False)
    batch_x=train_x[indices]
    batch_y=train_y[indices]
    return batch_x,batch_y

def LSTM_train(Data,Label,test_split):
    Data = Data.T
    Indices = np.arange(Data.shape[0])  # 随机打乱索引并切分训练集与测试集
    np.random.shuffle(Indices)

    train_sample_num = int(Data.shape[0]*test_split)

    train_x = Data[Indices[:train_sample_num]]
    train_y = Label[Indices[:train_sample_num]]
    test_x = Data[Indices[train_sample_num:]]
    test_y = Label[Indices[train_sample_num:]]



    print("1D-LSTM setup and initialize...")

    x = tf.placeholder(tf.float32, [None, 250])  # 定义placeholder数据入口
    x_ = tf.reshape(x, [-1, 250, 1])
    y_ = tf.placeholder(tf.float32, [None, 4])

    n_input = 1
    n_hidden = 1
    n_steps = 250
    n_classes = 4
    logits = LSTM_network(x_, n_input, n_hidden, n_steps, n_classes)

    learning_rate = 0.01
    batch_size = 16
    maxiters = 15000

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    tf.global_variables_initializer().run()




    print("1D-LSTM training and testing...")

    for i in range(maxiters):
        batch_x, batch_y = get_batch(train_x, train_y, batch_size)
        train_step.run(feed_dict={x: batch_x, y_: batch_y})
        if i % 500 == 0:
            loss = cost.eval(feed_dict={x: train_x, y_: train_y})
            print("Iteration %d/%d:loss %f" % (i, maxiters, loss))

    y_pred = logits.eval(feed_dict={x: test_x, y_: test_y})
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_y, axis=1)


    Acc = np.mean(y_pred == y_true)
    Conf_Mat = confusion_matrix(y_true, y_pred)  # 利用专用函数得到混淆矩阵
    Acc_N = Conf_Mat[0][0] / np.sum(Conf_Mat[0])
    Acc_V = Conf_Mat[1][1] / np.sum(Conf_Mat[1])
    Acc_R = Conf_Mat[2][2] / np.sum(Conf_Mat[2])
    Acc_L = Conf_Mat[3][3] / np.sum(Conf_Mat[3])
    return Acc, Acc_N, Acc_V,Acc_R,Acc_L, Conf_Mat