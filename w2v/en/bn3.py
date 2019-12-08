with open('fenci.txt','r',encoding='utf-8') as f:
    with open('input.txt','w',encoding='utf-8') as fo:
            cnt=0
            while(True):
                    a=f.readline()
                    if(len(a)==0):
                            break
                    a=a.replace(' ','X').replace('A\r','\r').replace('A\n','\n').replace('A',' ')
                    #print('1'+a+'1')
                    fo.write(a)
                    cnt+=1
                    #print(cnt)
                    if(cnt%10000==0):
                            print(cnt)
            fo.close()
    f.close()