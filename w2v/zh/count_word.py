import re

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


def spl(dd):
    if(re.search(RE,dd)):
        dd,zh,dec,pos = judge(dd)
        rev = 1
        if(pos==len(zh) and dec[0]==0 and checkzh(' '.join([i[0] for i in zh]).split())>15):
            dd,zh,dec,pos = judge(dd[::-1])
            rev = -1
        rch = '.'.join([zhget(zh[i]) for i in range(pos)])[::rev]
        ren = '.'.join([zhget(zh[i]) for i in range(pos,len(zh))])[::rev]
        return rch, ren
    else:
        return '', dd

cnt = 0
fd = '计算机科学的基础理论'
ret = []
for zz in range(34):
    with open('/mnt/disk1/tianxin/paper_' + str(zz) + '.txt', 'r') as f:
        linecnt = 0
        while(True):
            a = f.readline()
            if(len(a) == 0):
                break
            linecnt += 1
            if(linecnt % 200000 == 0):
                print(zz,linecnt,cnt)
            exec('b=' + a)
            tt = b['t']
            dd = b['d']
            pre = cnt
            if(tt):
                cnt+=(len(tt.split(fd))-1)
            if(dd):
                cnt+=(len(dd.split(fd))-1)
            if(cnt>pre):
                ret.append(tt)
                ret.append(dd)
    f.close()
print(cnt)
print(ret)