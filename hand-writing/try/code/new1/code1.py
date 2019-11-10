import numpy as np
import cv2
import struct

#数据存储路径
data_path = 'C:\\DataStore\\ML\\'

#获取数据列表
def init(load=False):
    train_img = []
    test_img = []
    if load:
        #从list.txt中直接读取列表
        with open("list.txt", "r") as f:
            [train_num, test_num] = f.readline().split()
            words = f.readline().split()
            data = [[] for i in range(len(words))]
            for i in range(int(train_num)):
                x = tuple(map(int, f.readline().split()))
                train_img.append(x)
                data[x[1]].append(x)
            for i in range(int(test_num)):
                x = tuple(map(int, f.readline().split()))
                test_img.append(x)
                data[x[1]].append(x)
        return words, train_img, test_img, data
    #从gnt文件中生成列表
    words = []
    data = []
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
                    data.append([])
                word = words.index(word)
                width = struct.unpack('<H', f.read(2))[0]
                height = struct.unpack('<H', f.read(2))[0]
                f.read(width * height)
                data[word].append((i, word, width, height, j + 10))
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
    return words, train_img, test_img, data

#读取图片
def read_image(x):
    with open(data_path + str(x[0]) + '-c.gnt', 'rb') as f:
        f.seek(x[4], 0)
        ia = np.fromfile(f, np.uint8, x[2] * x[3])
        ia.resize((x[3], x[2]))
        f.close()
    return ia

#图片预处理
def get_image(ia, img_size=84, add=False):
    #压缩大小到img_size*img_size
    ib = cv2.resize(ia, (img_size, img_size))
    #以192为阈值二值化图片
    ic = cv2.threshold(ib, 192, 1, cv2.THRESH_BINARY_INV)[1]
    #对数据增加随机干扰噪声
    if add and random.randint(0,1) == 1:
        xx = random.randint(0,75)
        for ix in range(xx):
            aa = random.randint(0,img_size)
            bb = random.randint(0,img_size)
            if ic[aa,bb] == 0:
                ic[aa,bb] = 1
            else:
                ic[aa,bb] = 0
    return ic[:,:,np.newaxis]
