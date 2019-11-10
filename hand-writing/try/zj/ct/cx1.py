import tensorflow as tf
import struct
import random
import time
import datetime
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout, BatchNormalization




def init(load=False):
    train_img = []
    test_img = []
    if load:
        with open("list.txt", "r") as f:
            [train_num, test_num] = f.readline().split()
            words = f.readline().split()
            for i in range(int(train_num)):
                train_img.append(tuple(map(lambda x: int(x), f.readline().split())))
            for i in range(int(test_num)):
                test_img.append(tuple(map(lambda x: int(x), f.readline().split())))
        return words, train_img, test_img
    words = []
    cnt = 0
    cc = []
    for i in range(1001, 1301):
        print(i)
        j = 0
        with open(data_path + str(i) + '-c.gnt', 'rb') as f:
            while True:
                sz = f.read(4)
                if len(sz) == 0:
                    break
                sz = struct.unpack('<I', sz)[0]
                word = f.read(2).decode('gb2312')
                if word not in words:
                    words.append(word)
                    cnt += 1
                    cc.append(0)
                word = words.index(word)
                width = struct.unpack('<H', f.read(2))[0]
                height = struct.unpack('<H', f.read(2))[0]
                f.read(width * height)
                if cc[word] < 200:
                    train_img.append((i, word, width, height, j + 10))
                else:
                    test_img.append((i, word, width, height, j + 10))
                cc[word] += 1
                j += sz
            f.close()
    with open('list.txt', "w") as f:
        f.write('%d %d\n' % (len(train_img), len(test_img)))
        for i in words:
            f.write(i + ' ')
        f.write('\n')
        for i in train_img + test_img:
            f.write('%d %d %d %d %d\n' % (i[0], i[1], i[2], i[3], i[4]))
        f.close()
    return words, train_img, test_img


def generate(data, size, step, r_len, words, add):
    random.shuffle(data)
    c = 0
    l = len(data)
    for i in range(step):
        sz = min(size, l - c)
        x = np.zeros((sz, 110, 110, 1))
        y = np.zeros((sz, r_len))
        for j in range(sz):
            with open(data_path + str(data[c][0]) + '-c.gnt', 'rb') as f:
                f.seek(data[c][4], 0)
                ia = np.fromfile(f, np.uint8, data[c][2] * data[c][3])
                f.close()
            ia.resize((data[c][3], data[c][2]))
            ib = cv2.resize(ia, (110, 110))
            ic = cv2.threshold(ib, 192, 1, cv2.THRESH_BINARY_INV)[1]
            #x[cnt] = cv2.dilate(ic, kernel)[:,:,np.newaxis]
            if add and random.randint(0,1) == 1:
                xx = random.randint(0,300)
                for ix in range(xx):
                    aa = random.randint(0,109)
                    bb = random.randint(0,109)
                    if ic[aa,bb] == 0:
                        ic[aa,bb] = 1
                    else:
                        ic[aa,bb] = 0
            x[j] = ic[:,:,np.newaxis]
            y[j][data[c][1]] = 1
            c += 1
        yield x, y



def get_cnn(r_len):
    model = Sequential()
    model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(110, 110, 1)))  # 54*54
    model.add(MaxPooling2D((2, 2)))  # 52*52
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
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(r_len, activation='relu'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    words, train_img, test_img = init(True)
    #train_img = train_img[:500]
    #test_img = test_img[:500]
    pre = test_img[0]
    r_len = len(words)
    model = get_cnn(r_len)
    load = False
    model.load_weights('cx1-14.h5')
    if load:
        
        return
    size = 500
    step_train = (len(train_img) + size - 1) // size
    step_test = (len(test_img) + size - 1) // size
    #train_img = data_input(train_img, r_len, False)
    #test_img = data_input(test_img, r_len, False)
    start = time.time()
    print('start:', datetime.datetime.now())
    for xx in range(15,21):
        if xx>12:
            model.fit_generator(generate(train_img, size, step_train, r_len, words, False), steps_per_epoch=step_train, epochs=1)
            model.save_weights('cx1-' + str(xx) + '.h5')
            print('TR:',xx,datetime.datetime.now(), 'use:', time.time() - start)
        if xx % 1 == 0:
            score = model.evaluate_generator(generate(train_img, size, step_train, r_len, words, False), steps=step_train, verbose=1)
            print('Train Loss:', score[0])
            print('Train Accuracy:', score[1])
            score = model.evaluate_generator(generate(test_img, size, step_test, r_len, words, False), steps=step_test, verbose=1)
            print('Test Loss:', score[0])
            print('Test Accuracy:', score[1])
            print('TE:',xx,datetime.datetime.now(), 'use:', time.time() - start)

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#data_path = './hw/'
data_path = 'C:\\DataStore\\ML\\'
main()
