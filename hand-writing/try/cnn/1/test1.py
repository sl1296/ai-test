import tensorflow as tf
import struct
import gc
import random
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Activation, Dropout
from keras.optimizers import SGD
from keras.initializers import RandomUniform



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


def get_image(x, add, words, train = True):
    if train:
        with open(data_path + str(x[0]) + '-c.gnt', 'rb') as f:
            f.seek(x[4], 0)
            mx = max(x[2], x[3])
            left = (mx - x[2]) // 2
            right = left + x[2]
            top = (mx - x[3]) // 2
            bottom = top + x[3]
            img = Image.new('1', (mx, mx))
            array = img.load()
            for a in range(mx):
                for b in range(mx):
                    val = 0
                    if a>=top and a<bottom and b>=left and b<right:
                        if ord(f.read(1)) < 255:
                            val = 1
                    array[b, a] = val
            f.close()
    else:
        img = Image.open(x).convert('1')
        plt.imshow(img)
        plt.show()
    #plt.imshow(img)
    #plt.show()
    '''        
    if random.randint(0,3) == 0:
        add = False
    if add:
        arg = random.randint(0,359)
        img = img.rotate(arg)
    '''
    img = img.resize((54, 54), resample=Image.LANCZOS)
    '''
    if add:
        array = img.load()
        cnt = random.randint(0,200)
        for i in range(cnt):
            aa=random.randint(0,53)
            bb=random.randint(0,53)
            if array[aa, bb] == 0:
                array[aa, bb] = 1
            else:
                array[aa, bb] = 0
    '''
    #plt.figure(words[x[1]])
    #plt.imshow(img)
    #plt.show()
    return np.array(img)[:, :, np.newaxis]


def generate(data, size, step, r_len, add, words):
    while True:
        random.shuffle(data)
        for i in range(step):
            sz = min(size, len(data) - i * size)
            gc.collect()
            x = np.zeros((sz, 54, 54, 1))
            y = np.zeros((sz, r_len))
            for j in range(sz):
                x[j] = get_image(data[i * size+ j], add, words)
                y[j][data[i * size + j][1]] = 1
            gc.collect()
            yield x, y


def get_cnn(r_len):
    model = Sequential()
    model.add(Convolution2D(2, (3, 3), activation='relu', input_shape=(54, 54, 1)))  # 54*54
    model.add(MaxPooling2D((2, 2)))  # 52*52
    model.add(Dropout(0.5))
    model.add(Convolution2D(4, (3, 3), activation='relu'))  # 26*26
    model.add(MaxPooling2D((2, 2)))  # 24*24
    model.add(Dropout(0.5))
    model.add(Convolution2D(8, (3, 3), activation='relu'))  # 12*12
    model.add(MaxPooling2D((2, 2)))  # 10*10
    model.add(Flatten())  # 5*5
    model.add(Dense(200, activation='relu'))
    model.add(Dense(r_len, activation='relu'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    words, train_img, test_img = init(True)
    r_len = len(words)
    model = get_cnn(r_len)
    load = False
    if load:
        model.load_weights('cnn.h5')
        ret = model.predict_classes(get_image('1.png',False,False)[np.newaxis,:,:,:])
        print(words[ret[0]])
        
        return
    size = 100
    step_train = (len(train_img) + size - 1) // size
    step_test = (len(test_img) + size - 1) // size
    for xx in range(50):
        model.fit_generator(generate(train_img, size, step_train, r_len, True, words), steps_per_epoch=step_train, epochs=1)
        model.save_weights('cnn-' + str(xx) + '.h5')
    score = model.evaluate_generator(generate(train_img, size, step_train, r_len, True, words), steps=step_train, verbose=1)
    print('Train Loss:', score[0])
    print('Train Accuracy:', score[1])
    score = model.evaluate_generator(generate(test_img, size, step_test, r_len, True, words), steps=step_test, verbose=1)
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])
    model.save_weights('cnn.h5')

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
data_path = 'C:\\DataStore\\ML\\'
main()
