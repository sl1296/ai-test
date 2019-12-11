import re
import time
from gensim.models import KeyedVectors
import jieba
import csv

RE = re.compile(u'[\u4e00-\u9fa5]',re.UNICODE)
null = None

def deckh(x):
    cs = 0
    for i in x:
        if(i=='('):
            cs+=1
        elif(i==')'):
            cs-=1
    if(cs!=0):
        return x,[]
    r=''
    info = []
    pre = -1
    prei = -1
    add = ''
    for i in range(len(x)):
        if(x[i]=='('):
            cs+=1
            if(cs==1):
                pre=len(r)
                if(i==0 or i>0 and x[i-1]!=')'):
                    prei=i
                add=x[i]
            elif(cs>1):
                add+=x[i]
            else:
                r+=x[i]
        elif(x[i]==')'):
            cs-=1
            if(cs>0):
                add+=x[i]
            elif(cs==0):
                add+=x[i]
                info.append((pre,add))
            else:
                r+=x[i]
        elif(cs<=0):
            if(cs==0 and x[i]=='.' and x[i-1]==')' and (prei>0 and x[prei-1]=='.' or prei==0)):
                info[-1] = (info[-1][0], info[-1][1] + '.')
            else:
                r+=x[i]
        else:
            add+=x[i]
    if(len(r)>0):
        if(r[-1]=='.'):
            r = r[:-1]
            info[-1] = (info[-1][0]-1, '.' + info[-1][1])
        return r,info
    else:
        return x,[]


def decyh(x):
    st = ''
    for i in x[0]:
        if(i=='"' or i=="'"):
            if(len(st)==0):
                st += i
            elif(len(st)==1):
                if(st[0]==i):
                    st = ''
                else:
                    st += i
            else:
                if(st[1]==i):
                    st = st[0]
                else:
                    break
    if(len(st)>0):
        return (x[0], [], x[1])
    r = ''
    rl = []
    add = ''
    pre = -1
    for i in range(len(x[0])):
        if(x[0][i]=='"' or x[0][i]=="'"):
            if(len(st)==0):
                st += x[0][i]
                pre = len(r)
                add = x[0][i]
            elif(len(st)==1):
                if(st[0]==x[0][i]):
                    st = ''
                    add += x[0][i]
                    rl.append((pre, add))
                else:
                    st += x[0][i]
                    add += x[0][i]
            else:
                st = st[0]
                add += x[0][i]
        elif(len(st)==0):
            r += x[0][i]
        else:
            add += x[0][i]
    if(len(r)==0):
        return (x[0], [], x[1])
    else:
        return (r, rl, x[1])


def nopoint(zh, pos):
    tmp = zh[pos][0].split()
    if(len(tmp)<5):
        return zh, pos
    avg = 0
    for i in range(1,len(tmp)):
        avg += len(tmp[i])
    if(avg/(len(tmp)-1)*2>len(tmp[0])):
        return zh, pos
    cc = 0
    p = len(tmp[0]) - 1
    while(p>=0 and ord(tmp[0][p])<128):
        p -= 1
    if(p<10):
        return zh, pos
    for i in tmp[0][:p+1]:
        if(ord(i)<128):
            cc += 1
    if(cc/p>0.2):
        return zh, pos
    lena = p+1
    cfa = [zh[pos][0][:p+1], [], []]
    cfb = [zh[pos][0][p+1:], [], []]
    for i in zh[pos][1]:
        if(i[0]<=p+1):
            cfa[1].append(i)
            lena += len(i[1])
        else:
            cfb[1].append(i)
    for i in zh[pos][2]:
        if(i[0]<=lena):
            cfa[2].append(i)
        else:
            cfb[2].append(i)
    return zh[:pos] + [cfa, cfb] + zh[pos+1:], pos+1


def checkzh(x):
    ce = 0
    for i in x:
        cc = 0
        for j in i:
            if(j>='a' and j<='z' or j>='A' and j<='Z'):
                cc += 1
        if(cc==len(i)):
            ce += 1
    return ce


def zhget(x):
    pre = 0
    r = ''
    tmp = ''
    for j in x[1]:
        tmp+=x[0][pre:j[0]]
        pre=j[0]
        tmp+=j[1]
    tmp+=x[0][pre:]
    pre = 0
    for j in x[2]:
        r+=tmp[pre:j[0]]
        pre=j[0]
        r+=j[1]
    r+=tmp[pre:]
    return r


def judge(ddd):
    ddd = ' '.join(ddd.replace('\n','').replace('。','.').replace('．','.').replace('（','(').replace('）',')').replace('’',"'").replace('‘',"'").replace('”','"').replace('“','"').split())
    ddd = ddd.split('.')
    ddd = '.'.join([i.strip() for i in ddd if(len(i.strip())>0)])
    zh, tmp = deckh(ddd)
    zh = zh.split('.')
    for i in range(len(zh)-1, 0, -1):
        if(len(zh[i-1])==0 or len(zh[i])==0):
            print(ddd)
            print(zh)
            print(tmp)
            aa=input()
        if(zh[i-1][-1]>='0' and zh[i-1][-1]<='9' and zh[i][0]>='0' and zh[i][0]<='9'):
            zh[i-1] = zh[i-1] + '.' + zh[i]
            zh[i] = ''
    zh = [[i, []] for i in zh if len(i)>0]
    now = 0
    sum = -1
    for i in tmp:
        while(now<len(zh) and sum<i[0]):
            sum+=(len(zh[now][0])+1)
            now+=1
        zh[now-1][1].append((i[0]-sum+len(zh[now-1][0]), i[1]))
    for i in range(len(zh)):
        zh[i] = decyh(zh[i])
    dec = [-1] * len(zh)
    cd = 0
    for i in range(len(zh)):
        cc=0
        j = zh[i][0].split()
        for k in range(len(j)):
            if(re.search(RE, j[k])):
                cc+=(len(j[k])+1)
            else:
                ccc = 0
                for ii in j[k]:
                    if(ii>='a' and ii<='z' or ii>='A' and ii<='Z'):
                        ccc += 1
                if(ccc*2<=len(j[k])):
                    cc += len(j[k])/2
        if(cc/len(zh[i][0])<=0.5):
            dec[cd] = i
            cd += 1
    pos = len(zh)
    while(True):
        if(cd>0):
            cd-=1
        else:
            break
        if(dec[cd]==pos-1):
            pos-=1
        else:
            break
    enp = ''.join([zh[i][0] for i in range(pos, len(zh))])
    numc = 0
    for i in enp:
        if(i>='0' and i<='9' or i=='.'):
            numc+=1
    if(len(enp)<45):
        pos=len(zh)
    if(pos<len(zh)):
        zh, pos = nopoint(zh, pos)
    if(pos==0 and len(zh)==1 and re.search(RE, zh[0][0])):
        pos = 1
    if(len(enp)<150 and (enp.find(':')!=-1 or enp.find('：')!=-1) and numc/len(enp)>0.2 and re.search(RE, enp)):
        pos = len(zh)
    ddd = '.'.join([zhget(i) for i in zh])
    return ddd,zh,dec,pos


def choose(universe, parts, start, end):
    if start < 0 or end < 0 or start >= len(universe) or end >= len(universe) \
            or not parts:
        return []
    c = {len(word): (word, i, j, k) for word, i, j, k in parts}
    w, i, j, k = c[max(c.keys())]
    partial_parts_left = list(filter(lambda x: x[1] < i and x[2] < i, parts))
    partial_parts_right = list(filter(lambda x: x[1] > j and x[2] > j, parts))
    left = choose(universe, partial_parts_left, start, i - 1)
    right = choose(universe, partial_parts_right, j + 1, end)
    return left + [(w, i, j, k)] + right


def tokk(text):
    text = [i.strip() for i in jieba.lcut(text.lower()) if len(i.strip())>0]
    inds = []
    now = ''
    for w in text:
        inds.append(len(now))
        now = ''.join([now, w])
    words = []
    parts = []
    for i in range(len(text) + 1):
        for j in range(i + 1, min(len(text) + 1, i + max_len)):
            w = ''.join(text[i: j])
            if w in dic:
                parts.append((w, i, j - 1, inds[i]))
    ret = choose(text, parts, 0, len(text)-1)
    r = []
    pos = 0
    for i in range(len(ret)):
        if(ret[i][1] > pos):
            for j in range(pos, ret[i][1]):
                if(text[j] in unk):
                    unk[text[j]] += 1
                else:
                    unk[text[j]] = 1
                r.append(text[j])
        pos = ret[i][2] + 1
        r.append(ret[i][0])
    return r


def zhfc(x):
    return ' '.join([i.replace(' ','X') for i in tokk(x) if len(i)>0])


unkid = 1
def saveunk(unk, zz):
    global unkid
    if(unkid==1):
        unkid=0
    else:
        unkid=1
    with open('unk'+str(zz)+'-'+str(unkid)+'.txt','w',encoding='utf-8') as f:
        f.write(str(time.time()-tm))
        f.write('\n')
        for i in unk:
            f.write(i.replace(' ','X'))
            f.write(' ')
            f.write(str(unk[i]))
            f.write('\n')
        f.close()


tm = time.time()
unk = []
with open('1.csv','r',encoding='utf-8') as f:
    unk += [i[1] for i in csv.reader(f)][1:]
    f.close()
with open('2.csv','r',encoding='utf-8') as f:
    unk += [i[1] for i in csv.reader(f)][1:]
    f.close()
with open('3.csv','r',encoding='utf-8') as f:
    unk += [i[8] for i in csv.reader(f)][1:]
    f.close()
path = 'x'
dic = KeyedVectors.load(path)
print(len(dic.wv.index2word))
dic = set.union(set(dic.wv.index2word), set(unk))
print(len(dic))
unk = {}
max_len = max([len(x) for x in dic])
print('start', time.time()-tm)
selec = int(input())
with open('ret' + str(selec) + '.txt','w',encoding='utf-8') as fo:
    for zz in range(selec, selec+1):
        with open('paper_' + str(zz) + '.txt', 'r') as f:
            linecnt = 0
            while(True):
                a = f.readline()
                if(len(a) == 0):
                    break
                linecnt += 1
                if(linecnt % 1000 == 0):
                    print(zz,linecnt,len(unk),time.time()-tm)
                if(linecnt % 20000 == 0):
                    saveunk(unk, zz)
                    print('unk saved as', unkid,time.time()-tm)
                exec('b=' + a)
                tt = b['t']
                dd = b['d']
                wt = True
                if(tt):
                    if(re.search(RE, tt)):
                        fo.write(zhfc(tt))
                        fo.write('\n')
                    else:
                        wt = False
                if(dd and re.search(RE, dd)):
                    dd,zh,dec,pos = judge(dd)
                    rev = 1
                    if(pos==len(zh) and dec[0]==0 and checkzh(' '.join([i[0] for i in zh]).split())>15):
                        dd,zh,dec,pos = judge(dd[::-1])
                        rev = -1
                    rch = '.'.join([zhget(zh[i]) for i in range(pos)])[::rev]
                    if(len(rch)>0):
                        if(rch[-1]!='.'):
                            rch = rch + '.'
                        fo.write(zhfc(rch))
                        fo.write('\n')
                        if(not wt):
                            fo.write(zhfc(tt))
                            fo.write('\n')
                    #ren = '.'.join([zhget(zh[i]) for i in range(pos,len(zh))])[::rev]
                    '''
                    eval = checkzh(' '.join([zh[i][0] for i in range(pos)]).split())
                    if(pos<len(zh) and '.'.join([zh[i][0] for i in range(pos,len(zh))])[:9]!='Objective' and (re.search(RE, ren) or eval>20) or pos==len(zh) and eval>5):
                        print(zz, linecnt, pos)
                        print(dec[:pos])
                        print('.'.join([zhget(i) for i in zh]))
                        print('-'*50)
                        print(rch)
                        print('-'*50)
                        print(ren)
                        print('-'*50)
                        b=input()
                    '''
            f.close()
        print('f', zz, linecnt, time.time()-tm)
        saveunk(unk, zz)
        print('unk saved as', unkid,time.time()-tm)
    fo.close()