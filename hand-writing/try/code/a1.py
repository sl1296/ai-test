import struct
import gc
import random
import time
import datetime
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import winsound


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
    if random.randint(0,3) == 0:
        add = False
    if add:
        arg = random.randint(0,359)
        img = img.rotate(arg)
    img = img.resize((54, 54), resample=Image.LANCZOS)
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
    #plt.figure(words[x[1]])
    #plt.imshow(img)
    #plt.show()
    return np.array(img)[:, :, np.newaxis]

'''
def show_image(x,y,words):
    img = Image.fromarray(x[:,:,0])
    tmp = -1
    for i in range(3755):
        if y[i] == 1:
            tmp = i
            break
    plt.figure(words[tmp])
    plt.imshow(img)
    plt.show()
'''

def generate(data, size, step, r_len, words, add):
    random.shuffle(data)
    c = 0
    l = len(data)
    for i in range(step):
        sz = min(size, l - c)
        x = np.zeros((sz, 1, 96, 96))
        y = np.zeros((sz, r_len))
        for j in range(sz):
            with open(data_path + str(data[c][0]) + '-c.gnt', 'rb') as f:
                f.seek(data[c][4], 0)
                ia = np.fromfile(f, np.uint8, data[c][2] * data[c][3])
                f.close()
            ia.resize((data[c][3], data[c][2]))
            ib = cv2.resize(ia, (96, 96))
            ic = cv2.threshold(ib, 192, 1, cv2.THRESH_BINARY_INV)[1]
            #x[cnt] = cv2.dilate(ic, kernel)[:,:,np.newaxis]
            if add and random.randint(0,1) == 1:
                xx = random.randint(0,75)
                for ix in range(xx):
                    aa = random.randint(0,53)
                    bb = random.randint(0,53)
                    if ic[aa,bb] == 0:
                        ic[aa,bb] = 1
                    else:
                        ic[aa,bb] = 0
            x[j] = ic[np.newaxis,:,:]
            y[j][data[c][1]] = 1
            '''
            plt.figure(words[data[c][1]])
            plt.imshow(ic)
            plt.show()
            '''
            c += 1
        yield x, y


def data_input(x, r_len, add):
    sz = len(x)
    r = [np.zeros((sz, 54, 54, 1)), np.zeros((sz, r_len))]
    for i in range(sz):
        r[0][i] = get_image(x[i], add, None)
        r[1][i][x[i][1]] = 1
        if i % 50 == 0:
            print('load:',i,'/',sz)
    print('load finish')
    return r


class Encode(nn.Module):
    def __init__(self, r_len):
        super(Encode, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.maxp1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.maxp2 = nn.MaxPool2d(2, return_indices=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.maxp3 = nn.MaxPool2d(2, return_indices=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.maxp4 = nn.MaxPool2d(2, return_indices=True)
        self.abc = nn.Sequential(
            nn.Linear(6*6*64, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000))

        self.den1 = nn.Sequential(
            nn.Linear(2000, r_len),
            nn.Softmax())
        
        self.cba = nn.Sequential(
            nn.Linear(2000, 6*6*64),
            nn.ReLU(),
            nn.BatchNorm1d(6*6*64))
        self.umaxp4 = nn.MaxUnpool2d(2)
        self.uconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.umaxp3 = nn.MaxUnpool2d(2)
        self.uconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.umaxp2 = nn.MaxUnpool2d(2)
        self.uconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.umaxp1 = nn.MaxUnpool2d(2)
        self.uconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.Tanh())
    def forward(self, x):
        x = self.conv1(x)
        x, id1 = self.maxp1(x)
        x = self.conv2(x)
        x, id2 = self.maxp2(x)
        x = self.conv3(x)
        x, id3 = self.maxp3(x)
        x = self.conv4(x)
        x, id4 = self.maxp4(x)
        x = x.view(x.size(0), -1)
        x = self.abc(x)

        #ret = self.den1(x)
        
        x = self.cba(x)
        x = x.view(x.size(0), 64, 6, 6)
        x = self.umaxp4(x, id4)
        x = self.uconv4(x)
        x = self.umaxp3(x, id3)
        x = self.uconv3(x)
        x = self.umaxp2(x, id2)
        x = self.uconv2(x)
        x = self.umaxp1(x, id1)
        x = self.uconv1(x)
        return x
 

def main():
    words, train_img, test_img = init(True)
    #train_img = train_img[:500]
    #test_img = test_img[:500]
    pre = test_img[0]
    r_len = len(words)
    
    size = 100
    step_train = (len(train_img) + size - 1) // size
    step_test = (len(test_img) + size - 1) // size

    model = Encode(r_len)
    model.cuda()
    #model.load_state_dict(torch.load('a1-0.pkl'))
    opt = torch.optim.Adam(model.parameters())
    lossx = nn.MSELoss()
    #lossy = nn.CrossEntropyLoss()
    
    start = time.time()
    print('start:', datetime.datetime.now())
    for xx in range(1001):
        
        closs = 0
        #clossx = 0
        #clossy = 0
        #cacc = 0
        step = 0
        print('Epoch', xx)
        pre = time.time()
        for tx, ty in generate(train_img, size, step_train, r_len, words, False):
            step += 1
            pos = round(step/(step_train/30))
            x = Variable(torch.from_numpy(tx).float()).cuda()
            #y = Variable(torch.from_numpy(ty).long()).cuda()
            rx = model(x)
            ls = lossx(rx, x)
            #lsy = lossy(ry, y)
            #ls = lsx + lsy
            #acc = (torch.argmax(ry, 1) == y).sum()
            #print(acc)
            opt.zero_grad()
            ls.backward()
            opt.step()
            #clossx += lsx.item()
            #clossy += lsy.item()
            closs += ls.item()
            eta = (time.time() - pre) / step * (step_train-step)
            print('\r%d/%d[' % (step, step_train)+'='*pos+'>'*min(1,30-pos)+'.'*(29-pos) + '] - ETA:%d:%02d - loss:%f'%(int(eta/60), int(eta%60), closs/step) +' '*30, end='')
        now = time.time()
        print('\r%d/%d[' % (step, step_train)+'='*pos+'>'*min(1,30-pos)+'.'*(29-pos) + '] - %ds %dms/step - loss:%f'%(round(now-pre), round((now-pre)*1000/step), closs/step) +' '*30)
        print(datetime.datetime.now(), 'use:', time.time() - start)
        torch.save(model.state_dict(), 'a1-'+str(xx)+'.pkl')
        
        if xx % 1 == 0:
            winsound.Beep(500, 500)
            closs = 0
            step = 0
            print('Train data:')
            pre = time.time()
            for tx, y in generate(train_img, size, step_train, r_len, words, False):
                step += 1
                pos = round(step/(step_train/30))
                x = Variable(torch.from_numpy(tx).float()).cuda()
                r = model(x)
                ls = lossx(r, x)
                if step == 1:
                    tr = r[:3,0:1,:,:].cpu().detach().numpy()
                    for num in range(3):
                        fig = plt.figure()
                        fig.suptitle('train')
                        plt.subplot(121)
                        plt.title('1')
                        plt.imshow(tx[num,0,:,:], cmap='gray')
                        plt.subplot(122)
                        plt.title('2')
                        plt.imshow(tr[num,0,:,:], cmap='gray')
                        plt.show()
                closs += ls.item()
                eta = (time.time() - pre) / step * (step_train-step)
                print('\r%d/%d[' % (step, step_train)+'='*pos+'>'*min(1,30-pos)+'.'*(29-pos) + '] - ETA:%d:%02d - loss:%f'%(int(eta/60), int(eta%60), closs/step) +' '*30, end='')
            now = time.time()
            print('\r%d/%d[' % (step, step_train)+'='*pos+'>'*min(1,30-pos)+'.'*(29-pos) + '] + %ds %dms/step + loss:%f'%(round(now-pre), round((now-pre)*1000/step), closs/step) +' '*30)
            winsound.Beep(500, 500)
            print('Test data:')
            closs = 0
            step = 0
            pre = time.time()
            for tx, y in generate(test_img, size, step_test, r_len, words, False):
                step += 1
                pos = round(step/(step_test/30))
                x = Variable(torch.from_numpy(tx).float()).cuda()
                r = model(x)
                ls = lossx(r, x)
                if step == 1:
                    tr = r[:3,0:1,:,:].cpu().detach().numpy()
                    for num in range(3):
                        fig = plt.figure()
                        fig.suptitle('test')
                        plt.subplot(121)
                        plt.title('1')
                        plt.imshow(tx[num,0,:,:], cmap='gray')
                        plt.subplot(122)
                        plt.title('2')
                        plt.imshow(tr[num,0,:,:], cmap='gray')
                        plt.show()
                closs += ls.item()
                eta = (time.time() - pre) / step * (step_test-step)
                print('\r%d/%d[' % (step, step_test)+'='*pos+'>'*min(1,30-pos)+'.'*(29-pos) + '] - ETA:%d:%02d - loss:%f'%(int(eta/60), int(eta%60), closs/step) +' '*30, end='')
            now = time.time()
            print('\r%d/%d[' % (step, step_train)+'='*pos+'>'*min(1,30-pos)+'.'*(29-pos) + '] + %ds %dms/step + loss:%f'%(round(now-pre), round((now-pre)*1000/step), closs/step) +' '*30)
            print(datetime.datetime.now(), 'use:', time.time() - start)
                
    
#data_path = './hw/'
data_path = 'C:\\DataStore\\ML\\'
main()
