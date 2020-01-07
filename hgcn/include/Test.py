import numpy as np
import scipy
delete = True

best_e = 0.0
best_r = 0.0

def get_hits(vec, test_pair, fname, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    ret = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        if(test_pair[i][1] == test_pair[rank[0]][1]):
            sg = 0
        else:
            sg = 1
        ret.append((test_pair[i][0], test_pair[rank[0]][1], sg))
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('Entity:')
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    global best_e
    if(top_lr[0]>best_e):
        best_e = top_lr[0]
        with open(fname + '_e.txt','w') as ff:
            for i in ret:
                ff.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
            ff.close()


        
def detect_coinc(test_pair, head, tail, ILL, fname):
    fname = fname + '.npy'
    if(delete):
        return np.load(fname)
    r2e = {}
    print('p1',len(test_pair))
    for ill in test_pair:
        if ill[0] not in r2e:
            r2e[ill[0]] = head[ill[0]] | tail[ill[0]]
        if ill[1] not in r2e:
            r2e[ill[1]] = head[ill[1]] | tail[ill[1]]
    print('p2')
    rpairs = {}
    test_pair = np.array(test_pair)
    left = test_pair[:, 0]
    right = test_pair[:, 1]
    print('p3',len(left),len(right),len(ILL))
    cnti = 0
    for i in left:
        cnti += 1
        print(cnti)
        for j in right:
            count = 0
            for e_1, e_2 in ILL:
                if e_1 in r2e[i] and e_2 in r2e[j]:
                    count = count + 1
            rpairs[(i, j)] = count / (len(r2e[i]) + len(r2e[j]))
    print('p4')
    coinc = []
    for row in left:
        list = []
        for col in right:
            list.append(rpairs[(row, col)])
        coinc.append(list)
    print('p5')
    coinc = np.array(coinc)
    np.save(fname, coinc)
    print(coinc)
    print('saved')
    return coinc


def get_hits_rel(vec, test_pair, coinc, fname, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    sim = sim - 20 * coinc
    top_lr = [0] * len(top_k)
    ret = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        if(test_pair[i][1] == test_pair[rank[0]][1]):
            sg = 0
        else:
            sg = 1
        ret.append((test_pair[i][0], test_pair[rank[0]][1], sg))
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('Relation:')
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    global best_r
    if(top_lr[0]>best_r):
        best_r = top_lr[0]
        with open(fname + '_r.txt','w') as ff:
            for i in ret:
                ff.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
            ff.close()
