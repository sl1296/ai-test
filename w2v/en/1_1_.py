import time
from gensim.models import KeyedVectors
import jieba
from nltk.tokenize import word_tokenize
'''
class en_process_func:
    def split_func(x):
        return word_tokenize(x)
    def merge_func(x):
        return ' '.join(x)
    def pre_process(x):
        return x.strip().lower()

class tokenizer:
    def __init__(self, dic, process_func):
        self.dic = dic
        self.split_func = process_func.split_func
        self.merge_func = process_func.merge_func
        self.pre_process = process_func.pre_process
        self.max_len = max([len(x) for x in self.dic])

    def choose(self, universe, parts, start, end):
        if start < 0 or end < 0 or start >= len(universe) or end >= len(universe) \
                or not parts:
            return []
        c = {len(word): (word, i, j, k) for word, i, j, k in parts}
        w, i, j, k = c[max(c.keys())]
        partial_parts_left = list(filter(lambda x: x[1] < i and x[2] < i, parts))
        partial_parts_right = list(filter(lambda x: x[1] > j and x[2] > j, parts))
        partial_res_left = self.choose(universe, partial_parts_left, start, i - 1)
        partial_res_right = self.choose(universe, partial_parts_right, j + 1, end)
        return partial_res_left + [(w, k)] + partial_res_right

    def tokenize(self, text, max_len=None):
        '#''Globally longest word first''#'
        if max_len is None:
            max_len = self.max_len
        text = self.pre_process(text)
        text = self.split_func(text)
        inds = []
        now = ""
        for w in text:
            inds.append(len(now))
            now = self.merge_func([now, w])
        words = []
        parts = []
        for i in range(len(text) + 1):
            for j in range(i + 1, min(len(text) + 1, i + max_len)):
                w = self.merge_func(text[i: j])
                if w in self.dic:
                    parts.append((w, i, j - 1, inds[i]))
        words += self.choose(text, parts, 0, len(text) - 1)
        return words

    def tokenize_greedy(self, text, max_len=None, oov=False):
        ''#'Locally longest word first, also good when doing from backward to frontward''#'
        if max_len is None:
            max_len = self.max_len
        text = self.pre_process(text)
        text = self.split_func(text)
        inds = []
        now = ""
        for w in text:
            inds.append(len(now))
            now = self.merge_func([now, w])
        text = text[::-1]
        inds = inds[::-1]
        words = []
        L = len(text)
        i = 0
        while i < L:
            flag = -1
            for j in range(min(L, i + max_len), i, -1):
                if self.merge_func(text[i:j][::-1]) in self.dic:
                    flag = j
                    break
            if flag != -1:
                words.append((self.merge_func(text[i:flag][::-1]), inds[flag-1]))
                i = flag
            else:
                if oov: words.append((text[i], inds[i]))
                i += 1
        return words[::-1]
'''
#tok = tokenizer(dic, en_process_func)

tm = time.time()

path = 'keywords_aminer_en'
#model = KeyedVectors.load(path)
#dic = set(model.wv.index2word)
#max_len = max([len(x) for x in dic])
#print(max_len)

unk = {}
unklist = []

def loadh():
    with open('1.txt','r',encoding='utf-8') as f:
        a = int(f.readline()[:-1])
        for i in range(a):
            a = f.readline()[:-1].split('A')
            unk[a[0]] = int(a[1])
        f.close()
            

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
    text = word_tokenize(text.strip().lower())
    inds = []
    now = ""
    for w in text:
        inds.append(len(now))
        now = ' '.join([now, w])
    words = []
    parts = []
    for i in range(len(text) + 1):
        for j in range(i + 1, min(len(text) + 1, i + max_len)):
            w = ' '.join(text[i: j])
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

def readdata():
    with open('0.csv','r',encoding='utf-8') as f:
        a=f.readline()
        #print(a)
        cnt = 0
        while(True):
            a = f.readline()
            #print(a)
            if(len(a) == 0):
                break
            st = a.find('","')
            if(st != -1):
                #docs.append(a[st+3:-2])
                cnt += 1
            if(cnt % 10000 == 0):
                print('read cnt =', cnt, 'TIME:', time.time()-tm)
            #if(cnt == 300000):
            #    print(a[st+3:-2])
            #    break
        f.close()
    print('read finish, cnt =', cnt, 'TIME:', time.time()-tm)
    #print(docs) 
 
def fenci():
    with open('0.csv','r',encoding='utf-8') as f1:
        with open('1.txt','a',encoding='utf8') as f2:
            f1.readline()
            cnt = 0
            while(True):
                a = f1.readline()
                if(len(a) == 0):
                    break
                st = a.find('","')
                if(st != -1):
                    cnt += 1
                    if (cnt <= 1500000):
                        continue
                    for i in tokk(a[st+3:-2]):
                        f2.write(i)
                        f2.write('A')
                    f2.write('\n')
                    if (cnt % 500 == 0):
                        print('cal cnt=', cnt, 'TIME:', time.time()-tm)
                    
                    #if(cnt == 1500000):
                    #    break
            #f2.write(str(cnt))
            print('cal end cnt=', cnt, 'TIME:', time.time()-tm)
            f2.close()
        f1.close()
    print('unk =', len(unk))
    for i in unk:
        unklist.append((i, unk[i]))
    print('change type finish TIME:', time.time()-tm)
    unklist.sort(key=lambda x:(-x[1], x[0]))
    print('sort finish TIME:', time.time()-tm)
    with open('1.txt','w',encoding='utf-8') as f:
        f.write(str(len(unk)))
        f.write('\n')
        for i in range(len(unklist)):
            f.write(unklist[i][0])
            f.write('A')
            f.write(str(unklist[i][1]))
            f.write('\n')
            if(i % 10000 == 9999):
                print('save x =', i, 'TIME:', time.time()-tm)
        f.close()
#loadh()
print('load finish unklen =', len(unk), 'TIME:', time.time()-tm)
readdata()
#fenci()