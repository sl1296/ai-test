from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def fj(imgpre,img,vxs=0,vxe=0,h=True):
    x = img.shape[0]
    y = img.shape[1]
    if h:
        fj = y // 70
        dis = 30
        s = [0 for i in range(x)]
        for i in range(x):
            for j in range(y):
                s[i] += img[i][j]
    else:
        fj = 5
        dis = 30
        s = [0 for i in range(y)]
        for i in range(x):
            for j in range(y):
                s[j] += img[i][j]
        x = y
    print(s)
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
    ret = []
    for i in range(0,len(ch),2):
        if h:
            ret.append((imgpre[ch[i]:ch[i+1]+1],img[ch[i]:ch[i+1]+1],ch[i],ch[i+1]))
        else:
            ret.append((imgpre[:,ch[i]:ch[i+1]+1],vxs,vxe,ch[i],ch[i+1],img[:,ch[i]:ch[i+1]+1]))
        #plt.imshow(ret[-1][0])
        #plt.show()
    return ret


def fz(img):
    x = img.shape[0]
    y = img.shape[1]
    ret = []
    return ret
        
    
out = cv2.imread('5.jpg', 0)
plt.imshow(out,cmap='gray')
plt.show()
ret, th = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imgpre = cv2.adaptiveThreshold(out, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,51,5)
plt.imshow(imgpre,cmap='gray')
plt.show()
img = cv2.fastNlMeansDenoising(imgpre,h=90)
plt.imshow(img,cmap='gray')
plt.show()
'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
img = cv2.dilate(img, kernel)
plt.imshow(img)
plt.show()
'''
#ret, th = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
hh = fj(imgpre,img)
zz = []
for i in hh:
    zz += fj(i[0],i[1],i[2],i[3],False)
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
for i in range(len(zz)):
    cv2.imwrite('r'+str(i)+'.png',zz[i][0])
plt.imshow(out,cmap='gray')
plt.show()
