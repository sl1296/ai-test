import tensorflow as tf
import random
import time
import datetime
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Convolution2D, MaxPooling2D, Dropout, BatchNormalization
import argparse
from code1 import init, read_image, get_image

#获取训练或测试数据的迭代器
def generate(data, size, step, r_len, words, add):
    #随机打乱数据顺序
    random.shuffle(data)
    c = 0
    l = len(data)
    for i in range(step):
        sz = min(size, l - c)
        x = np.zeros((sz, 110, 110, 1))
        y = np.zeros((sz, r_len))
        #每次更新参数取sz组数据
        for j in range(sz):
            #读取图片数据
            x[j] = get_image(read_image(data[c]), 110, add)
            #设置数据答案标签
            y[j][data[c][1]] = 1
            c += 1
        yield x, y

#获取全连接神经网络模型
def get_den(r_len,xx,drop):
    model = Sequential()
    model.add(Flatten(input_shape=(110, 110, 1)))
    model.add(Dense(units=1000, activation=xx))
    model.add(Dropout(drop))
    model.add(BatchNormalization())
    model.add(Dense(units=200, activation=xx))
    model.add(Dropout(drop))
    model.add(BatchNormalization())
    model.add(Dense(units=100, activation=xx))
    model.add(Dropout(drop))
    model.add(BatchNormalization())
    model.add(Dense(units=r_len, activation=xx))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#获取卷积神经网络模型
def get_cnn(r_len,aa,bb,cc,dd,ee,xx):
    model = Sequential()
    model.add(Convolution2D(aa, (3, 3), activation=xx, input_shape=(110, 110, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(bb, (3, 3), activation=xx))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(cc, (3, 3), activation=xx))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(dd, (3, 3), activation=xx))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(ee, activation=xx))
    model.add(BatchNormalization())
    model.add(Dense(r_len, activation=xx))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def sup(md, drop, activation, noise, load, image):
    #获取数据列表和字符种类数
    words, train_img, test_img, _ = init(True)
    r_len = len(words)
    #获取神经网络模型
    if md == 'd1':
        model = get_den(r_len,activation,drop)
    elif md == 'c1':
        model = get_cnn(r_len,2,4,8,16,200,activation)
    elif md == 'c2':
        model = get_cnn(r_len,16,16,32,64,200,activation)
    elif md == 'c3':
        model = get_cnn(r_len,16,64,64,64,1000,activation)
    elif md == 'c4':
        model = get_cnn(r_len,64,96,192,384,1000,activation)
    #每次更新参数的数据量，遍历全部训练或测试数据的次数
    size = 500
    step_train = (len(train_img) + size - 1) // size
    step_test = (len(test_img) + size - 1) // size
    #测试图片
    if load > 0:
        model.load_weights(md + '-' + activation + '-' + str(drop) + '-' + str(noise) + '-' + str(load) + '.h5')
        img = get_image(cv2.imread('1.png',cv2.IMREAD_GRAYSCALE), 110)[np.newaxis,:,:,:]
        ret = model.predict_classes(img)
        print(words[ret[0]])
        return words[ret[0]]
    #记录时间
    start = time.time()
    print('start:', datetime.datetime.now())
    for xx in range(1,21):
        #遍历一次训练集训练模型
        model.fit_generator(generate(train_img, size, step_train, r_len, words, noise), steps_per_epoch=step_train, epochs=1)
        #保存模型参数
        model.save_weights(md + '-' + activation + '-' + str(drop) + '-' + str(noise) + '-' + str(xx) + '.h5')
        #记录时间
        print('TR:',xx,datetime.datetime.now(), 'use:', time.time() - start)
        #在训练集上测试模型准确率
        score = model.evaluate_generator(generate(train_img, size, step_train, r_len, words, False), steps=step_train, verbose=1)
        print('Train Loss:', score[0])
        print('Train Accuracy:', score[1])
        #在测试集上测试模型准确率
        score = model.evaluate_generator(generate(test_img, size, step_test, r_len, words, False), steps=step_test, verbose=1)
        print('Test Loss:', score[0])
        print('Test Accuracy:', score[1])
        #记录时间
        print('TE:',xx,datetime.datetime.now(), 'use:', time.time() - start)

#参数解析
parser = argparse.ArgumentParser()
parser.add_argument("-m","--model",type = str, default = 'd1')
parser.add_argument("-d","--drop",type = float, default = 0.0)
parser.add_argument("-a","--activation",type = str, default = 'relu')
parser.add_argument("-n","--noise",type = bool, default = False)
parser.add_argument("-l","--load",type = int, default = 0)
parser.add_argument("-i","--image",type = str, default = '')
args = parser.parse_args()
md = args.model
drop = args.drop
activation = args.activation
noise = args.noise
load = args.load
image = args.image
#tensorflow的GPU设置
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#训练神经网络
sup(md, drop, activation, noise, load, image)
