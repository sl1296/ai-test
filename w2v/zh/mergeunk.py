unk = {}
for i in range(34):
    id = 0
    ta = 0
    tb = 0
    with open('unk' + str(i) + '-0.txt', 'r') as f:
        ta = float(f.readline()[:-1])
        f.close()
    with open('unk' + str(i) + '-1.txt', 'r') as f:
        tb = float(f.readline()[:-1])
        f.close()
    if(ta<tb):
        id = 1
    with open('unk' + str(i) + '-' +str(id) + '.txt', 'r') as f:
        a = f.readline()
        while(True):
            a = f.readline()
            if(len(a)==0):
                break
            p = a.split()
            if(p[0] not in unk):
                unk[p[0]] = int(p[1])
            else:
                unk[p[0]] += int(p[1])
        f.close()
    print('f',i,len(unk))
unklist = [(i,unk[i]) for i in unk]
unklist.sort(key=lambda x:(-x[1], x[0]))
with open('unkall.txt','w') as f:
    for i in unklist:
        f.write(i[0])
        f.write(' ')
        f.write(str(i[1]))
        f.write('\n')
    f.close()