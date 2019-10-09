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
dropout_rate = 0.7
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


# 读取下一个 batch 的数据,默认不打乱.  用 for 循环
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


def image_show(image, label, pred, num, sign):
    miou_v1 = 0
    err = abs(pred - label[0, :, :, 0])
    temp = 1 - (np.sum(err) / (img_rows * img_cols))
    fig = plt.figure(figsize=(15, 4))

    #     miou_v1 = mean_iou(label, pred)
    if sign == 0:
        fig.suptitle("train image " + str(num) + ", acc: " + str(float('%.4f' % temp)))
    #                                                                 " miou: " + str(float("%.4f"%miou_v1)))
    else:
        fig.suptitle("test image " + str(num) + ", acc: " + str(float('%.2f' % temp)))
    #                                                                 " miou: " + str(float("%.4f"%miou_v1)))
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


def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation="relu", name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation="relu", name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)

    return x


def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    nb_filter = [32, 64, 128, 256, 256]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])

    return model


def train():
    start_time = datetime.datetime.now()
    
    train_txt_name = 'train.txt'
    test_txt_name = 'test.txt'
    
    model = Nest_Net(img_rows=512, img_cols=512, color_type=1)
    opt = Adam()
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[mean_iou])
    # model.summary()
    #model.load_weights('600.h5')
    
    for epoch in range(0,601):
        for xs, ys in next_batch(train_txt_name, batch_size, True):
            # history = model.fit(xs,ys, batch_size = 1, epochs = 1, verbose = 0)
            # print(history.history)
            history = model.train_on_batch(xs, ys)

        if epoch % 10 == 0:
            model.save_weights(str(epoch) + '.h5')
            for xs_test, ys_test in next_batch(test_txt_name, 1, False):
                score = model.evaluate(xs_test, ys_test, batch_size=1, verbose=0)
            print("%3.0f epoch,  tran loss: %.5f,  train  miou : %.5f" % (epoch, history[0], history[1]) + ", test " + str(score))
            

    k = 0
    for xs_test, ys_test in next_batch(test_txt_name, 1, False):
        k = k + 1
        pred = model.predict(xs_test)
        #         print("----", pred[0].sum())
        for i in range(512):
            for j in range(512):
                if pred[0, i, j, 0] > 0.5:
                    pred[0, i, j, 0] = 1
                else:
                    pred[0, i, j, 0] = 0
        image_show(xs_test, ys_test, pred[0, :, :, 0], k, 1)
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
        img_arr = np.squeeze(img_arr)  # 三维图片
        img2 = nib.load('.\\label\\segmentation-' + str(j) + '.nii')
        img_arr2 = img2.get_fdata()
        img_arr2 = np.squeeze(img_arr2)  # 三维图片
        for i in range(img_arr.shape[2]):
            img_name = '.\\image\\'+ str(j) + '_' + str(i) + '.png'
            #image = img_arr[:, :, i] * 255 / 4 + 0.3
            #cv2.imwrite(img_name, np.transpose(image))  # 要旋转一下
            img_name2 = '.\\label\\'+ str(j) + '_' + str(i) + '.png'
            #image2 = img_arr2[:, :, i] * 255 / 4 + 0.3
            #cv2.imwrite(img_name2, np.transpose(image2))  # 要旋转一下
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
