import random
import time
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision.transforms as transforms
import scipy as sp
import scipy.stats
import argparse
from code1 import init,read_image,get_image

#计算准确率的平均值和误差
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

#特征降维网络定义
class CNNEncoder(nn.Module):
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
        return out

#关系判断网络定义
class RelationNetwork(nn.Module):
    def __init__(self):
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
        self.fc1 = nn.Linear(64*3*3, 8)
        self.fc2 = nn.Linear(8, 1)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

#初始化网络参数
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

#训练和测试模型B时获取数据，train为True表示训练，False表示测试
def get_data(data, class_num, sample_num, batch_num, train, transform):
    if train:
        a = random.sample(range(2000), class_num)
    else:
        a = random.sample(range(2000, 3755), class_num)
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
    return samples, batches, batch_labels

#训练模型A时获取数据
def get_data2(data, transform):
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

def trs(tt):
    #图片正则化函数
    normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    #读取数据列表
    words, train_img, test_img, data = init(True)
    r_len = len(words)
    #特征降维网络定义
    feature_encoder = CNNEncoder()
    #关系判断网络定义
    relation_network = RelationNetwork()
    #初始化网络参数
    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)
    #存储网络到GPU
    feature_encoder.cuda()
    relation_network.cuda()
    #使用adam方法更新学习率
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=0.001)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=0.001)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)
    #记录时间
    start = time.time()
    print('start:', datetime.datetime.now())
    if tt == 't':
        #测试模型A+B
        for z in range(10):
            test_data = [random.sample(range(len(data[i])), 6) for i in range(r_len)]
            item = random.sample(range(r_len), 50)
            acc = 0
            cc = 0
            acc2 = 0
            for i in item:
                #加载模型A参数，计算结果
                feature_encoder.load_state_dict(torch.load('A-fe-150000.pkl'))
                relation_network.load_state_dict(torch.load('A-rn-150000.pkl'))
                cc += 1
                ret = [0 for j in range(r_len)]
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
                #输出模型A准确率
                print(i, ret.index(max(ret)), ret[i], cnt, acc/cc, end='|| ')
                #计算模型A相似度最高的10个类型
                rmx = list(range(10))
                rmx.sort(reverse=True, key=lambda zz:ret[zz])
                for j in range(10, r_len):
                    if ret[j] > ret[rmx[9]]:
                        rmx[9] = j
                        rmx.sort(reverse=True, key=lambda zz:ret[zz])
                #加载模型B参数，计算10个类型的结果
                feature_encoder.load_state_dict(torch.load('B-fe-200000.pkl'))
                relation_network.load_state_dict(torch.load('B-rn-200000.pkl'))
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
                if pre == i:
                    acc2 += 1
                #输出模型B的准确率
                print(pre, acc2/cc)
            #输出一次测试50个字的结果，记录时间
            accuracies.append(acc2/cc)
            print('TE:', z, 'acc:', acc/cc, acc2/cc, datetime.datetime.now(), 'use:', time.time() - start)
            print(datetime.datetime.now(), 'use:', time.time() - start)
        #输出准确率的平均值和误差
        test_accuracy, h = mean_confidence_interval(accuracies)
        print("\ntest accuracy:",test_accuracy,"h:",h)
        return
    for xx in range(1, 200000):
        feature_encoder_scheduler.step(xx)
        relation_network_scheduler.step(xx)
        #训练，随机抽取一组数据
        if tt == 'a':
            samples, batches, batch_labels = get_data2(data, transform)
        elif tt == 'b':
            samples, batches, batch_labels = get_data(data, CLASS_NUM, SAMPLE_NUM, BATCH_NUM, True, transform)
        #计算样本数据特征
        sample_features = feature_encoder(Variable(samples).cuda())
        if tt == 'b':
            sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM, 64, 19, 19)
            sample_features = torch.sum(sample_features, 1).squeeze(1)
        #计算查询数据特征
        batch_features = feature_encoder(Variable(batches).cuda())
        #组合查询数据和样本数据的特征，计算相关度
        if tt == 'a':
            relation_pairs = torch.cat((sample_features, batch_features), 1)
            relations = relation_network(relation_pairs)
        elif tt == 'b':
            sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM*CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            relation_pairs = torch.cat((sample_features_ext, batch_features_ext),2).view(-1, 128, 19, 19)
            relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
        #计算loss函数
        mse = nn.MSELoss().cuda()
        if tt == 'a':
            batch_labels = batch_labels.cuda()
            acc = 0
            for i in range(A*2):
                if batch_labels[i][0] == 1 and relations[i][0] > 0.99:
                    acc += 1
                if batch_labels[i][0] == 0 and relations[i][0] < 0.01:
                    acc += 1
            loss = mse(relations, batch_labels)
        elif tt == 'b':
            one_hot_labels = Variable(torch.zeros(BATCH_NUM*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1).long(), 1).cuda())
            loss = mse(relations, one_hot_labels)
        #更新参数
        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()
        #输出loss，记录时间
        if xx % 5 == 0:
            if tt == 'a':
                print("episode:",xx,"loss",loss.data.item(), "acc:", acc/300)
            elif tt == 'b':
                print("episode:",xx,"loss",loss.data.item())
            print(datetime.datetime.now(), 'use:', time.time() - start)
        #模型A保存网络参数
        if xx % 1000 == 0 and tt == 'a':
            torch.save(feature_encoder.state_dict(), 'A-fe-'+str(xx)+'.pkl')
            torch.save(relation_network.state_dict(), 'A-rn-'+str(xx)+'.pkl')
        #模型B保存网络参数并测试
        if xx % 1000 == 0 and tt == 'b':
            torch.save(feature_encoder.state_dict(), 'B-fe-'+str(xx)+'.pkl')
            torch.save(relation_network.state_dict(), 'B-rn-'+str(xx)+'.pkl')
            print("Testing...")
            accuracies = []
            #测试600次计算准确率的平均值和误差
            for i in range(600):
                #获取测试数据
                samples, batches, batch_labels = get_data(data, CLASS_NUM, SAMPLE_NUM, BATCH_NUM, False, transform)
                #计算样本数据特征
                sample_features = feature_encoder(Variable(samples).cuda())
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM, 64, 19, 19)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                #计算查询数据特征
                batch_features = feature_encoder(Variable(batches).cuda())
                #组合样本和查询数据特征，计算相关度
                sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM*CLASS_NUM, 1, 1, 1, 1)
                batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
                relation_pairs = torch.cat((sample_features_ext, batch_features_ext),2).view(-1, 128, 19, 19)
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
                #计算正确率
                _,predict_labels = torch.max(relations.data,1)
                predict_labels = predict_labels.cpu()
                batch_labels = batch_labels.long()
                sz = BATCH_NUM * CLASS_NUM
                rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(sz)]
                accuracies.append(np.sum(rewards) / sz)
                #输出测试进度
                if i % 20 == 0:
                    num = i // 20
                    print('\r[' + '>'*num + '='*(30-num) + ']', end='')
            #计算输出测试结果，记录时间
            test_accuracy, h = mean_confidence_interval(accuracies)
            print("\ntest accuracy:",test_accuracy,"h:",h)
            print(datetime.datetime.now(), 'use:', time.time() - start)

#参数解析
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",type = str, default = 't')
args = parser.parse_args()
tt = args.type
image = args.image
#类别数，每类的样本数和查询数
CLASS_NUM = 10
SAMPLE_NUM = 5
BATCH_NUM = 10
A=100
#训练模型
trs(tt)
