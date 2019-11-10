import tensorflow as tf
import struct
import gc
import random
import cv2
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.initializers import RandomUniform



def init(load=False):
    if load:
        with open("lst.txt", "r") as f:
            [train, test] = f.readline().split()
            words = f.readline().split()
            f.close()
        return words, int(train), int(test)
    words = []
    train = 0
    test = 0
    for i in range(1001, 1301):
        print(i)
        with open(data_path + str(i) + '-c.gnt', 'rb') as f:
            while True:
                if len(f.read(4)) == 0:
                    break
                if i < 1241:
                    train += 1
                else:
                    test += 1
                word = f.read(2).decode('gb2312')
                if word not in words:
                    words.append(word)
                width = struct.unpack('<H', f.read(2))[0]
                height = struct.unpack('<H', f.read(2))[0]
                f.read(width * height)
            f.close()
    with open('lst.txt', "w") as f:
        f.write('%d %d\n' % (train, test))
        for i in words:
            f.write(i + ' ')
        f.write('\n')
        f.close()
    return words, train, test


def generate(s, e, size, step, r_len, words, mp, add):
    fs = list(range(s,e))
    random.shuffle(fs)
    x = np.zeros((size, 110, 110, 1))
    y = np.zeros((size, r_len))
    #kernel = np.ones((3, 3), np.uint8)
    cnt = 0
    for i in fs:
        with open(data_path + str(i) + '-c.gnt', 'rb') as f:
            while True:
                if len(f.read(4)) == 0:
                    break
                word = f.read(2).decode('gb2312')
                width = struct.unpack('<H', f.read(2))[0]
                height = struct.unpack('<H', f.read(2))[0]
                ia = np.fromfile(f, np.uint8, width * height)
                ia.resize((height, width))
                ib = cv2.resize(ia, (110, 110))
                ic = cv2.threshold(ib, 192, 1, cv2.THRESH_BINARY_INV)[1]
                #x[cnt] = cv2.dilate(ic, kernel)[:,:,np.newaxis]
                x[cnt] = ic[:,:,np.newaxis]
                '''
                plt.figure(word)
                plt.imshow(x[cnt,:,:,0])
                plt.show()
                '''
                y[cnt][mp[word]] = 1
                cnt += 1
                if cnt == size:
                    yield x, y
                    x = np.zeros((size, 110, 110, 1))
                    y = np.zeros((size, r_len))
                    cnt = 0
            f.close()
    yield x[0:cnt,:,:,:], y[0:cnt,:]


def get_cnn(r_len):
    model = Sequential()
    model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(110, 110, 1)))  # 110*110
    model.add(MaxPooling2D((2, 2)))  # 108*108
    model.add(BatchNormalization())
    model.add(Convolution2D(16, (3, 3), activation='relu'))  # 54*54
    model.add(MaxPooling2D((2, 2)))  # 52*52
    model.add(BatchNormalization())
    model.add(Convolution2D(32, (3, 3), activation='relu'))  # 26*26
    model.add(MaxPooling2D((2, 2)))  # 24*24
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), activation='relu'))  # 12*12
    model.add(MaxPooling2D((2, 2)))  # 10*10
    model.add(BatchNormalization())
    model.add(Flatten())  # 5*5
    model.add(Dense(r_len, activation='relu'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model


def main():
    words, train, test = init(True)
    r_len = len(words)
    mp = {}
    for i in range(r_len):
        mp[words[i]] = i
    model = get_cnn(r_len)
    use = False
    #model.load_weights('')
    if use:
        model.load_weights('')
        ret = model.predict_classes(get_image('1.png', False, words, False)[np.newaxis,:,:,:])
        print(words[ret[0]])
        return
    size = 64
    step_train = (train + size - 1) // size
    step_test = (test + size - 1) // size
    start = time.time()
    print(datetime.datetime.now())
    for xx in range(0,501):
        model.fit_generator(generate(1001, 1241, size, step_train, r_len, words, mp, False), steps_per_epoch=step_train, epochs=1)
        model.save_weights('cnn-C+-' + str(xx) + '.h5')
        print(datetime.datetime.now(), 'use:', time.time() - start)
        if xx % 5 == 0:
            score = model.evaluate_generator(generate(1001, 1241, size, step_train, r_len, words, mp, False), steps=step_train, verbose=1)
            print('Train Loss:', score[0])
            print('Train Accuracy:', score[1])
            score = model.evaluate_generator(generate(1241, 1301, size, step_test, r_len, words, mp, False), steps=step_test, verbose=1)
            print('Test Loss:', score[0])
            print('Test Accuracy:', score[1])
            print(datetime.datetime.now(), 'use:', time.time() - start)



config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#data_path = './hw/'
data_path = 'C:\\DataStore\\ML\\'
main()
