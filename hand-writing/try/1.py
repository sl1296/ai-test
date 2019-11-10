import struct
import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
word = {}
cword = 0
ctrain = 0
ctest = 0
mxw = 0
mxh = 0
ts = 1001
tm = 1002
te = 1003
for z in range(ts,tm):
    print('z=',z)
    with open('C:\\DataStore\\ML\\' + str(z) + '-c.gnt','rb') as f:
        while True:
            tmp = f.read(4)
            if(len(tmp) == 0):
                break
            xx = f.read(2).decode('gb2312')
            if(xx not in word):
                word[xx] = cword
                cword += 1
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            if(mxw < width):
                mxw = width
            if(mxh < height):
                mxh = height
            f.read(width * height)
            ctrain += 1
        f.close()
for z in range(tm,te):
    print('z=',z)
    with open('C:\\DataStore\\ML\\' + str(z) + '-c.gnt','rb') as f:
        while True:
            tmp = f.read(4)
            if(len(tmp) == 0):
                break
            xx = f.read(2).decode('gb2312')
            if(xx not in word):
                word[xx] = cword
                cword += 1
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            if(mxw < width):
                mxw = width
            if(mxh < height):
                mxh = height
            f.read(width * height)
            ctest += 1
        f.close()
print('cword:',cword,'train_num:',ctrain,'test_num:',ctest,'height:',mxh,'width:',mxw)
train_x = np.zeros((ctrain,mxh*mxw),dtype='float32')
train_y = np.zeros((ctrain,cword),dtype='float32')
test_x = np.zeros((ctest,mxh*mxw),dtype='float32')
test_y = np.zeros((ctest,cword),dtype='float32')
cnt = 0
for z in range(ts,tm):
    print('z=',z)
    with open('C:\\DataStore\\ML\\' + str(z) + '-c.gnt','rb') as f:
        while True:
            tmp = f.read(4)
            if(len(tmp) == 0):
                break
            tag_code = f.read(2).decode('gb2312')
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            up = (mxh - height) // 2
            left = (mxw - width) // 2
            down = up + height
            right = left + width
            for i in range(0,mxh):
                for j in range(0,mxw):
                    if(i >= up and i < down and j >= left and j < right):
                        tmp = struct.unpack('<B',f.read(1))[0]
                        train_x[cnt][i*mxw+j] = (255 - tmp) / 255
            train_y[cnt][word[tag_code]] = 1
            cnt += 1
        f.close()
cnt = 0
for z in range(tm,te):
    print('z=',z)
    with open('C:\\DataStore\\ML\\' + str(z) + '-c.gnt','rb') as f:
        while True:
            tmp = f.read(4)
            if(len(tmp) == 0):
                break
            tag_code = f.read(2).decode('gb2312')
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            up = (mxh - height) // 2
            left = (mxw - width) // 2
            down = up + height
            right = left + width
            for i in range(0,mxh):
                for j in range(0,mxw):
                    if(i >= up and i < down and j >= left and j < right):
                        tmp = struct.unpack('<B',f.read(1))[0]
                        test_x[cnt][i*mxw+j] = (255 - tmp) / 255
            test_y[cnt][word[tag_code]] = 1
            cnt += 1
        f.close()
print(mxw, mxh)
model=Sequential()
model.add(Dense(input_dim=mxw*mxh,units=400,activation='relu'))
model.add(Dense(units=400,activation='relu'))
model.add(Dense(units=cword,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=50, epochs=4)
score = model.evaluate(train_x, train_y)
print('\nTrain Acc:', score)
score = model.evaluate(test_x, test_y)
print('\nTest Acc:', score)
model_json = model.to_json()
with open('model.json', 'w') as f:
        f.write(model_json)
        f.close()
model.save_weights('model.h5')
