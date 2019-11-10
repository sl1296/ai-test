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

gs = [(i,j) for i in range(2001) for j in range(i+1,2001)]
gsp = 0
vis = {}
def get_data(data, class_num, sample_num, batch_num, xx):
    normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    if xx != -1:
        global gs
        global vis
        global gsp
        if gsp == -1:
            random.shuffle(gs)
            gsp = len(gs)-1
            print('reset GS')
        a = []
        for i in range(0, class_num//2):
            while gsp > -1 and vis.get(gs[gsp]) != None:
                gsp -= 1
            if gsp == -1:
                break
            if gs[gsp][0] not in a:
                a.append(gs[gsp][0])
            if gs[gsp][1] not in a:
                a.append(gs[gsp][1])
            gsp -= 1
        while len(a) < class_num:
            tmp = random.randint(0, 2000)
            if tmp not in a:
                a.append(tmp)
        for i in range(class_num):
            for j in range(i+1, class_num):
                vis[(min(a[i],a[j]),max(a[i],a[j]))] = 1
        #a = random.sample(range(2001), class_num)
        dic = dict(zip(a, range(class_num)))
        b = [list(zip([i]*(sample_num+batch_num), random.sample(range(len(data[i])), sample_num+batch_num))) for i in a]
        samples = torch.zeros(class_num*sample_num, 1, 84, 84)
        batches = torch.zeros(class_num*batch_num, 1, 84, 84)
        batch_labels = torch.zeros(class_num*batch_num)
        cc = 0
        for i in b:
            for j in i[:sample_num]:
                samples[cc] = transform(get_image(read_image(data[j[0]][j[1]])))
                cc += 1
        c = [j for i in b for j in i[sample_num:]]
        random.shuffle(c)
        for i in range(len(c)):
            batches[i] = transform(get_image(read_image(data[c[i][0]][c[i][1]])))
            batch_labels[i] = dic[data[c[i][0]][c[i][1]][1]]
    else:
        a = random.sample(range(2001, 3755), class_num)
        dic = dict(zip(a, range(class_num)))
        b = [list(zip([i]*(sample_num+batch_num), random.sample(range(len(data[i])), sample_num+batch_num))) for i in a]
        samples = torch.zeros(class_num*sample_num, 1, 84, 84)
        batches = torch.zeros(class_num*batch_num, 1, 84, 84)
        batch_labels = torch.zeros(class_num*batch_num)
        cc = 0
        for i in b:
            for j in i[:sample_num]:
                samples[cc] = transform(get_image(read_image(data[j[0]][j[1]])))
                cc += 1
        c = [j for i in b for j in i[sample_num:]]
        #random.shuffle(c)
        for i in range(len(c)):
            batches[i] = transform(get_image(read_image(data[c[i][0]][c[i][1]])))
            batch_labels[i] = dic[data[c[i][0]][c[i][1]][1]]
        
        
    return samples, batches, batch_labels
    


def main():
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
    '''
    if os.path.exists('abc'):
        feature_encoder.load_state_dict(torch.load('feature'))
        print("load feature encoder success")
    if os.path.exists('abc'):
        relation_network.load_state_dict(torch.load('relation'))
        print("load relation network success")
    '''
    print('Training...')
    feature_encoder.load_state_dict(torch.load('b1+30-fe-5350.pkl'))
    relation_network.load_state_dict(torch.load('b1+30-rn-5350.pkl'))
    start = time.time()
    print('start:', datetime.datetime.now())
    for xx in range(5351, 1000001):
        
        samples, batches, batch_labels = get_data(data, CLASS_NUM, SAMPLE_NUM, BATCH_NUM, xx)
        feature_encoder_scheduler.step(xx)
        relation_network_scheduler.step(xx)
        sample_features = feature_encoder(Variable(samples).cuda())
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM, 64, 19, 19)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda())

        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM*CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext),2).view(-1, 128, 19, 19)
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

        mse = nn.MSELoss().cuda()
        one_hot_labels = Variable(torch.zeros(BATCH_NUM*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1).long(), 1).cuda())
        loss = mse(relations, one_hot_labels)

        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()
        if xx%500 == 0:
            print("episode:",xx,"loss",loss.data.item())
            print(datetime.datetime.now(), 'use:', time.time() - start)
            torch.save(feature_encoder.state_dict(), 'b1+30-fe-'+str(xx)+'.pkl')
            torch.save(relation_network.state_dict(), 'b1+30-rn-'+str(xx)+'.pkl')
        
        if xx%500 == 0:
            print("Testing...")
            accuracies = []
            for i in range(600):
                samples, batches, batch_labels = get_data(data, CLASS_NUM, SAMPLE_NUM, BATCH_NUM, -1)
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
            test_accuracy, h = mean_confidence_interval(accuracies)
            print("\ntest accuracy:",test_accuracy,"h:",h)
            print(datetime.datetime.now(), 'use:', time.time() - start)

                
            
                
    return
    '''
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
    ''' 
    
#data_path = './hw/'
data_path = 'C:\\DataStore\\ML\\'
img_size = 84
CLASS_NUM = 30
SAMPLE_NUM = 5
BATCH_NUM = 2
main()
