with open('ret0.txt','a',encoding='utf-8') as fo:
    for i in range(1,34):
        with open('ret'+str(i)+'.txt','r',encoding='utf-8') as f:
            while(True):
                a=f.readline()
                if(len(a)==0):
                    break
                fo.write(a)
            f.close()
        with open('ret'+str(i)+'.txt','w',encoding='utf-8') as fx:
            fx.close()
    fo.close()