import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout

import nibabel as nib
import gc
import random

import numpy as np
import os
import datetime
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob


smooth = 1.
dropout_rate = 0.3
act = "relu"
batch_size = 1
epoch_num = 1
img_rows = 512
img_cols = 512


def read_image(temp_names):
    image_list_arr = []
    label_list_arr = []
    for content in temp_names:
        #
        image = plt.imread('./image/'+content[:-1]) / 255
        #print('C:\\DataStore\\image\\'+content[:-1])
        image_list_arr.append(image)

        #
        label = plt.imread('./label/'+content[:-1]) / 255
        for i in range(512):
            for j in range(512):
                if label[i,j] > 0:
                    label[i,j] = 1
        #print('C:\\DataStore\\label\\'+content[:-1])
        #plt.imshow(plt.imread('C:\\DataStore\\label\\'+content[:-1]))
        #plt.show()
        label_list_arr.append(label)
    image_list_arr = np.array(image_list_arr, dtype=np.float32)[:,:,:,np.newaxis]
    label_list_arr = np.array(label_list_arr, dtype=np.int32)[:,:,:,np.newaxis]

    return (image_list_arr, label_list_arr)


# 璇诲彇涓嬩竴涓?batch 鐨勬暟鎹?榛樿涓嶆墦涔?  鐢?for 寰幆
def next_batch(txtname, batch_size, shuffle=False):
    f = open(txtname, "r")
    contexts = np.array(f.readlines())
    f.close()
    indices = np.arange(len(contexts[0:]))
    if shuffle == True:
        np.random.shuffle(indices)

    for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
        exceprt = indices[start_idx: start_idx + batch_size]
        images, labels = read_image(contexts[exceprt])
        gc.collect()
        yield images, labels


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)




def image_show(image, label, pred, num, sign, score):
    miou_v1 = 0
    err = abs(pred - label[0, :, :, 0])
    temp = 1 - (np.sum(err) / (img_rows * img_cols))
    fig = plt.figure(figsize=(15, 4))

    #miou_v1 = mean_iou(label, pred)
    if sign == 0:
        fig.suptitle("train image " + str(num) + ", loss: " + str(score[0]) + ", miou: " + str(score[1]))
    else:
        fig.suptitle("test image " + str(num) + ", loss: " + str(score[0]) + ", miou: " + str(score[1]))
    plt.subplot(141)
    plt.title("original image " + str(num))
    plt.imshow(image[0, :, :, 0], cmap='gray')

    plt.subplot(142)
    plt.title("labels " + str(num))
    plt.imshow(label[0, :, :, 0], cmap="gray")

    plt.subplot(143)
    plt.title("prediction image " + str(num))
    plt.imshow(pred, cmap="gray")

    plt.subplot(144)
    plt.title("error image " + str(num))
    plt.imshow(err, cmap="gray")
    plt.show()


def Nest_Net(img_rows = 512, img_cols = 512):
    inputs = Input((img_rows, img_cols,1))
    # 缃戠粶缁撴瀯瀹氫箟
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print ("pool1 shape:",pool1.shape)
    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print ("pool2 shape:",pool2.shape)

    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print ("pool3 shape:",pool3.shape)

    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(dropout_rate)(conv5)



    up6 = Conv2DTranspose(256, (2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
         
    up7 = Conv2DTranspose(128,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    merge7 =concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
         
    up8 = Conv2DTranspose(64,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 =concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
         
    up9 = Conv2DTranspose(32,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 =concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)


    '''
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    '''
    model = Model(input = inputs, output = conv10)
    
    return model


def train():
    start_time = datetime.datetime.now()
    
    train_txt_name = 'train.txt'
    test_txt_name = 'test.txt'
    
    model = Nest_Net()
    opt = Adam()
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[mean_iou])
    # model.summary()
    model.load_weights('unet2-70.h5')
    
    for epoch in range(0,0):
        his = [0, 0]
        cnt = 0
        for xs, ys in next_batch(train_txt_name, batch_size, True):
            # history = model.fit(xs,ys, batch_size = 1, epochs = 1, verbose = 0)
            # print(history.history)
            history = model.train_on_batch(xs, ys)
            his[0] += history[0]
            his[1] += history[1]
            cnt += 1
        print('epoch %d:train loss:%f, train miou:%f' % (epoch, his[0]/cnt, his[1]/cnt))
        if epoch % 5 == 0:
            sc = [0, 0]
            cnt = 0
            model.save_weights('unet2-' + str(epoch) + '.h5')
            for xs_test, ys_test in next_batch(test_txt_name, 1, False):
                score = model.evaluate(xs_test, ys_test, batch_size=1, verbose=0)
                sc[0] += score[0]
                sc[1] += score[1]
                cnt += 1
            print("test loss: %.5f, test miou: %.5f" % (sc[0]/cnt, sc[1]/cnt))
            

    k = 0
    for xs_test, ys_test in next_batch(test_txt_name, 1, False):
        k = k + 1
        score = model.evaluate(xs_test, ys_test, batch_size=1, verbose=0)
        pred = model.predict(xs_test)
        #         print("----", pred[0].sum())
        for i in range(512):
            for j in range(512):
                if pred[0, i, j, 0] > 0.5:
                    pred[0, i, j, 0] = 1
                else:
                    pred[0, i, j, 0] = 0
        image_show(xs_test, ys_test, pred[0, :, :, 0], k, 1, score)
        # plt.imshow(pred[0,:,:,0] * 255, cmap = "gray")
        # plt.show()
    end_time = datetime.datetime.now()
    print("start time:  ", start_time)
    print("end time:    ", end_time)


def init():
    train = []
    test = []
    for j in range(0, 28):
        print(j)
        img = nib.load('.\\image\\volume-' + str(j) + '.nii')
        img_arr = img.get_fdata()
        img_arr = np.squeeze(img_arr)  # 涓夌淮鍥剧墖
        img2 = nib.load('.\\label\\segmentation-' + str(j) + '.nii')
        img_arr2 = img2.get_fdata()
        img_arr2 = np.squeeze(img_arr2)  # 涓夌淮鍥剧墖
        for i in range(img_arr.shape[2]):
            img_name = '.\\image\\'+ str(j) + '_' + str(i) + '.png'
            #image = img_arr[:, :, i] * 255 / 4 + 0.3
            #cv2.imwrite(img_name, np.transpose(image))  # 瑕佹棆杞竴涓?            img_name2 = '.\\label\\'+ str(j) + '_' + str(i) + '.png'
            #image2 = img_arr2[:, :, i] * 255 / 4 + 0.3
            #cv2.imwrite(img_name2, np.transpose(image2))  # 瑕佹棆杞竴涓?
            if j < 14:
                train.append((img_name,img_name2))
            else:
                test.append((img_name,img_name2))
    random.shuffle(train)
    random.shuffle(test)
    with open('train.txt','w') as f:
        for i in train[:100]:
            f.write('%s %s\n' % (i[0], i[1]))
        f.close()
    with open('test.txt','w') as f:
        for i in test[:20]:
            f.write('%s %s\n' % (i[0], i[1]))
        f.close()


def tmp():
    with open('train.txt','r') as f:
        xx = f.readlines()
        for i in xx:
            x = i.split()
            os.system('copy ' + x[0] + ' C:\\DataStore\\image\\' + x[0][8:])
            os.system('copy ' + x[1] + ' C:\\DataStore\\label\\' + x[0][8:])
        f.close()
    with open('test.txt','r') as f:
        xx = f.readlines()
        for i in xx:
            x = i.split()
            os.system('copy ' + x[0] + ' C:\\DataStore\\image\\' + x[0][8:])
            os.system('copy ' + x[1] + ' C:\\DataStore\\label\\' + x[0][8:])
        f.close()


config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
if __name__ == '__main__':
    #init()
    #tmp()
    #with tf.device('/cpu:0'):
    train()
