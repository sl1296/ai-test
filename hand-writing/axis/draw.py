import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def sigmoid(x):
    return 1/(1+2.7182818284590452**(-x))
def relu(x):
    y = x.copy()
    print(y.shape)
    for i in range(len(y)):
        if y[i]<0:
            y[i]=0
    return y
def selu(x):
    y = x.copy()
    print(y.shape)
    for i in range(len(y)):
        if y[i]<0:
            y[i]=1.050107*(1.673263*(2.7182818284590452**y[i]-1))
        else:
            y[i]=1.050107*y[i]
    return y

mpl.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
x = np.linspace(-10,10,10000)
y = relu(x)

ax.tick_params(labelsize=25)
ax.spines['top'].set_color('none')  
ax.spines['right'].set_color('none')  

ax.xaxis.set_ticks_position('bottom')  
ax.spines['bottom'].set_position(('data',0))  
ax.set_xticks([-10,-5,0,5,10])  
ax.yaxis.set_ticks_position('left')  
ax.spines['left'].set_position(('data',0))  
ax.set_yticks([5,10])  

plt.plot(x,y,label = 'relu',linestyle='-',color='black',linewidth=5)

plt.savefig('relu.png')
