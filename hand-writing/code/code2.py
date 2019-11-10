import cv2
import numpy as np
import matplotlib.pyplot as plt

#path为图片路径，expa为True使用膨胀后找连通块方法，为False使用投影切分方法
path = '1.jpg'
expa = True

#投影切分方法
def fj(imgpre, img, vxs=0, vxe=0, h=True):
    x = img.shape[0]#图片高度
    y = img.shape[1]#图片宽度
    if h:
        #切分行投影统计
        fj = y // 70
        dis = 30
        s = [0 for i in range(x)]
        for i in range(x):
            for j in range(y):
                s[i] += img[i][j]
    else:
        #切分列投影统计
        fj = 5
        dis = 30
        s = [0 for i in range(y)]
        for i in range(x):
            for j in range(y):
                s[j] += img[i][j]
        x = y
    #根据阈值fj寻找分割点
    rec = []
    ch = []
    now = False
    for i in range(x):
        if s[i] > fj:
            if now == False:
                rec.append(i)
                now = True
        else:
            if now == True:
                rec.append(i)
                now = False
    if now:
        rec.append(x)
    #合并间距小的区间，删除不正确的区间
    ps = rec[0]
    pe = rec[1]
    for i in range(2,len(rec),2):
        if rec[i] - pe < dis:
            pe = rec[i+1]
        else:
            if pe-ps > 70:
                ch.append(ps)
                ch.append(pe)
            ps = rec[i]
            pe = rec[i+1]
    ch.append(ps)
    ch.append(pe)
    #返回分割结果的图片和坐标
    ret = []
    for i in range(0,len(ch),2):
        if h:
            ret.append((imgpre[ch[i]:ch[i+1]+1],img[ch[i]:ch[i+1]+1],ch[i],ch[i+1]))
        else:
            ret.append((imgpre[:,ch[i]:ch[i+1]+1],vxs,vxe,ch[i],ch[i+1]))
    return ret

#找连通块切分方法
def fj2(preimg,img):
    x = img.shape[0]#图片高度
    y = img.shape[1]#图片宽度
    #广度优先搜索算法寻找连通块，记录坐标范围
    vis = [[0 for j in range(y)] for i in range(x)]
    ret = []
    for i in range(x):
        for j in range(y):
            #遍历所有像素点
            if vis[i][j] == 0 and img[i,j] == 1:
                #对于没有标记过的前景点找连通块
                cnt = 0
                xmin = i
                xmax = i
                ymin = j
                ymax = j
                bfs = []
                bfs.append((i,j))
                vis[i][j]=1
                cnt = 1
                cc = 1
                p = 0
                while cc>0:
                    now = bfs[p]
                    cc-=1
                    p+=1
                    for z in ((-1,0),(1,0),(0,-1),(0,1)):
                        na = now[0]+z[0]
                        nb = now[1]+z[1]
                        if na>=0 and na<x and nb>=0 and nb<y and vis[na][nb]==0 and img[na,nb]==1:
                            bfs.append((na,nb))
                            xmin=min(xmin,na)
                            xmax=max(xmax,na)
                            ymin=min(ymin,nb)
                            ymax=max(ymax,nb)
                            vis[na][nb]=1
                            cc+=1
                            cnt+=1
                #删除较小的噪声连通块
                if cnt>5500:
                    ret.append((0,xmin,xmax,ymin,ymax))
    return ret

#读取图片
out = cv2.imread(path, 0)
#自适应二值化
imgpre = cv2.adaptiveThreshold(out, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 5)
#去噪声
img = cv2.fastNlMeansDenoising(imgpre,h=90)
#显示二值化后的图片
plt.imshow(img,cmap='gray')
plt.show()
if expa:
    #膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    img = cv2.dilate(img, kernel)
    #按连通块切分文字
    zz = fj2(imgpre,img)
else:
    #按投影切分行
    hh = fj(imgpre,img)
    #按投影切分列
    zz = []
    for i in hh:
        zz += fj(i[0],i[1],i[2],i[3],False)
#结果标记方框
for i in zz:
    print(i)
    for j in range(i[1],i[2]+1):
        for k in range(20):
            out[j,i[3]+k]=0
            out[j,i[4]-k]=0
    for j in range(i[3],i[4]+1):
        for k in range(20):
            out[i[1]+k,j]=0
            out[i[2]-k,j]=0
#显示切分结果
plt.imshow(out,cmap='gray')
plt.show()
