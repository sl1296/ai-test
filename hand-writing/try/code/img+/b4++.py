import struct
import random
import time
import datetime
import winsound
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#import skimage
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision.transforms as transforms
import scipy as sp
import scipy.stats



def init(load=False):
    train_img = []
    test_img = []
    if load:
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


def read_image(x):
    with open(data_path + str(x[0]) + '-c.gnt', 'rb') as f:
        f.seek(x[4], 0)
        ia = np.fromfile(f, np.uint8, x[2] * x[3])
        ia.resize((x[3], x[2]))
        f.close()
    return ia


def get_image(ia, add=False):
    ib = cv2.resize(ia, (img_size, img_size))
    ic = cv2.threshold(ib, 192, 1, cv2.THRESH_BINARY_INV)[1]
    #x[cnt] = cv2.dilate(ic, kernel)[:,:,np.newaxis]
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


def generate(data, size, step, r_len, words, add):
    random.shuffle(data)
    c = 0
    l = len(data)
    for i in range(step):
        sz = min(size, l - c)
        x = np.zeros((sz, 1, img_size, img_size))
        y = np.zeros((sz, r_len))
        for j in range(sz):
            x[j] = get_image(read_image(data[c]), add)
            y[j][data[c][1]] = 1
            '''
            plt.figure(words[data[c][1]])
            plt.imshow(ic)
            plt.show()
            '''
            c += 1
        yield x, y

#RN
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def get_data(data, transform):
    ret = []
    if random.sample(range(15), 1)[0] < 13:
        c = random.sample(range(2000), A*2-A//3+1)
        b = random.sample(range(len(data[c[0]])), A//3+1)
        ret = []
        for i in range(1, len(b)):
            ret.append((c[0], b[0], c[0], b[i], 1))
        for i in range(1, len(c)):
            d = random.sample(range(len(data[c[i]])), 1)[0]
            ret.append((c[0], b[0], c[i], d, 0)) 
    else:
        for i in range(A):
            x = random.sample(range(2000), 1)[0]
            a = random.sample(range(len(data[x])), 2)
            ret.append((x, a[0], x, a[1], 1))
        for i in range(A):
            x = random.sample(range(2000), 2)
            a = random.sample(range(len(data[x[0]])), 1)[0]
            b = random.sample(range(len(data[x[1]])), 1)[0]
            ret.append((x[0], a, x[1], b, 0))
    random.shuffle(ret)
    samples = torch.zeros(A*2, 1, 84, 84)
    batches = torch.zeros(A*2, 1, 84, 84)
    batch_labels = torch.zeros(A*2, 1)
    for i in range(A*2):
        samples[i] = transform(get_image(read_image(data[ret[i][0]][ret[i][1]])))
        batches[i] = transform(get_image(read_image(data[ret[i][2]][ret[i][3]])))
        batch_labels[i][0] = ret[i][4]
            
    return samples, batches, batch_labels

A=100
def main():
    normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    words, train_img, test_img, data = init(True)
    r_len = len(words)
    
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(64, 8)
    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)
    feature_encoder.cuda()
    relation_network.cuda()
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=0.001)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=0.001)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)
    feature_encoder.load_state_dict(torch.load('b4-fe-151800.pkl'))
    relation_network.load_state_dict(torch.load('b4-rn-151800.pkl'))
    '''
    if os.path.exists('abc'):
        feature_encoder.load_state_dict(torch.load('feature'))
        print("load feature encoder success")
    if os.path.exists('abc'):
        relation_network.load_state_dict(torch.load('relation'))
        print("load relation network success")
    '''
    print('Training...')
    start = time.time()
    print('start:', datetime.datetime.now())
    for xx in range(151800,500001):
        '''
        samples, batches, batch_labels = get_data(data, transform)
        feature_encoder_scheduler.step(xx)
        relation_network_scheduler.step(xx)
        sample_features = feature_encoder(Variable(samples).cuda())
        batch_features = feature_encoder(Variable(batches).cuda())
        relation_pairs = torch.cat((sample_features, batch_features),1)
        relations = relation_network(relation_pairs)
        
        batch_labels = batch_labels.cuda()
        acc = 0
        for i in range(A*2):
            if batch_labels[i][0] == 1 and relations[i][0] > 0.99:
                acc += 1
            if batch_labels[i][0] == 0 and relations[i][0] < 0.01:
                acc += 1
        mse = nn.MSELoss().cuda()
        loss = mse(relations, batch_labels)

        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()
        if xx%20 == 0:
            print("episode:",xx,"loss",loss.data.item(), "acc:", acc/300)
            print(datetime.datetime.now(), 'use:', time.time() - start)
        if xx%200 == 0:
            torch.save(feature_encoder.state_dict(), 'b4-fe-'+str(xx)+'.pkl')
            torch.save(relation_network.state_dict(), 'b4-rn-'+str(xx)+'.pkl')
        '''
        if xx%1 == 0:
            print("Testing...")
            accuracies = []
            '''
            for i in range(600):
                samples, batches, batch_labels = get_data(data, CLASS_NUM, SAMPLE_NUM, BATCH_NUM, -1, transform)
                sample_features = feature_encoder(Variable(samples).cuda())
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM, 64, 19, 19)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                batch_features = feature_encoder(Variable(batches).cuda())
                sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM*CLASS_NUM, 1, 1, 1, 1)
                batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
                relation_pairs = torch.cat((sample_features_ext, batch_features_ext),2).view(-1, 128, 19, 19)
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
                _,predict_labels = torch.max(relations.data,1)
                predict_labels = predict_labels.cpu()
                batch_labels = batch_labels.long()
                sz = BATCH_NUM*CLASS_NUM
                rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(sz)]
                accuracies.append(np.sum(rewards) / sz)
                if i % 20 == 0:
                    num = i//20
                    print('\r['+'>'*num + '='*(30-num)+']',end='')
            '''
            for z in range(1):
                test_data = [random.sample(range(len(data[i])), 6) for i in range(r_len)]
                #item = random.sample(range(r_len), 50)
                item = range(1630,3755)
                acc = 0
                cc = 0
                acc2 = 0
                acc3 = 0
                for i in item:
                    feature_encoder.load_state_dict(torch.load('b4-fe-151800.pkl'))
                    relation_network.load_state_dict(torch.load('b4-rn-151800.pkl'))
                    cc += 1
                    ret = [0 for j in range(r_len)]
                    outp = read_image(data[i][test_data[i][-1]])
                    '''
                    plt.imshow(outp,cmap='gray')
                    plt.show()
                    '''
                    for j in range(0, r_len, 25):
                        sz = min(25, r_len-j)
                        batches = torch.zeros(sz*5, 1, 84, 84)
                        batches[0] = transform(get_image(read_image(data[i][test_data[i][-1]])))
                        for k in range(1,sz*5):
                            batches[k] = batches[0]
                        samples = torch.zeros(sz*5, 1, 84, 84)
                        for k in range(sz):
                            for l in range(5):
                                samples[k*5+l] = transform(get_image(read_image(data[j+k][test_data[j+k][l]])))
                        sample_features = feature_encoder(Variable(samples).cuda())
                        batch_features = feature_encoder(Variable(batches).cuda())
                        relation_pairs = torch.cat((sample_features, batch_features),1)
                        relations = relation_network(relation_pairs)
                        for k in range(sz):
                            for l in range(5):
                                ret[j+k] += float(relations.data[k*5+l][0])
                    if ret.index(max(ret)) == i:
                        acc += 1
                    cnt = 0
                    for j in ret:
                        if j > ret[i]:
                            cnt += 1
                    print(i, ret.index(max(ret)), ret[i], cnt, acc/cc, '||', words[i], words[ret.index(max(ret))],end='||')
                    
                    rmx = list(range(10))
                    rmx.sort(reverse=True, key=lambda zz:ret[zz])
                    for j in range(10, r_len):
                        if ret[j] > ret[rmx[9]]:
                            rmx[9] = j
                            rmx.sort(reverse=True, key=lambda zz:ret[zz])
                    '''
                    for j in rmx:
                        print(j, words[j], ret[j]/5, '|| ',end ='')
                    print()
                    '''
                    feature_encoder.load_state_dict(torch.load('b1-fe-201500.pkl'))
                    relation_network.load_state_dict(torch.load('b1-rn-201500.pkl'))
                    samples = torch.zeros(50, 1, 84, 84)
                    for j in range(10):
                        for k in range(5):
                            samples[j*5+k] = transform(get_image(read_image(data[rmx[j]][test_data[rmx[j]][k]])))
                    batches = torch.zeros(1, 1, 84, 84)
                    batches[0] = transform(get_image(read_image(data[i][test_data[i][-1]])))
                    sample_features = feature_encoder(Variable(samples).cuda())
                    sample_features = sample_features.view(10, 5, 64, 19, 19)
                    sample_features = torch.sum(sample_features, 1).squeeze(1)
                    batch_features = feature_encoder(Variable(batches).cuda()).repeat(10, 1, 1, 1)
                    relation_pairs = torch.cat((sample_features, batch_features),1)
                    relations = relation_network(relation_pairs)
                    now = 0
                    pre = -1
                    for j in range(10):
                        if relations[j][0] > now:
                            now = relations[j][0]
                            pre = rmx[j]
                        #print(rmx[j], words[rmx[j]], float(relations[j][0]),'|| ',end='')
                    if pre == i:
                        acc2 += 1
                    #print()
                    print(pre, acc2/cc,'||',words[pre])
                    
                #accuracies.append(acc/(r_len-1000))
            #test_accuracy, h = mean_confidence_interval(accuracies)
            #print("\ntest accuracy:",test_accuracy,"h:",h)
            print(datetime.datetime.now(), 'use:', time.time() - start)

    
#data_path = './hw/'
data_path = 'C:\\DataStore\\ML\\'
img_size = 84
main()
