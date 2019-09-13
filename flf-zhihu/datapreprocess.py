# -*- coding: utf-8 -*-
# 数据预处理，目标就是制作训练数据，
# 首先是问题数据预处理，三个部分，将标题向量、描述向量、话题向量三个向量保存成numpy
# 或者我可以先跑一个最简单的版本，只用一个描述向量进行一个分类。
import numpy as np

question_map={}
member_map={}
word_map={}
topic_map={}
# getlable,然后吧lable存成文件
def getlable(filename):

    ret = []
    with open(filename,'r',encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            ret.append(line[len(line)-1])
    return np.array(ret)


# 用data7获取训练数据，问题，和用户都存成numpy，先拿问题
def analysisdata(filename):
    '''
    测试数据集：
    问题标题长度：词最长38、字最长125、字-1数量0、词-1数量1362、词平均长度6.6
    字分布：[0, 13, 17, 85, 402, 2439, 4791, 11511, 22498, 38499, 63047, 79965, 91493, 99150, 103575, 103373, 101484,
    95878, 89670, 81737, 73083, 67088, 62095, 56277, 50861, 47087, 41295, 37631, 33722, 30931, 27773, 25710, 23019,
    20694, 18984, 17233, 15964, 14858, 13787, 13099, 12156, 11373, 10890, 10569, 10522, 10596, 10651, 11058, 11820,
    13381, 20254, 16325, 46, 34, 16, 14, 14, 16, 12, 13, 9, 7, 13, 5, 7, 5, 8, 9, 2, 4, 1, 2, 6, 4, 1, 3, 0, 3, 2,
    3, 1, 0, 2, 1, 0, 1, 1, 2, 1, 0, 2, 0, 3, 0, 0, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    词分布：[1362, 17139, 85674, 193636, 261277, 266374, 230487, 186524, 144188, 109203, 82803, 62251, 48885, 37757,
    29919, 23619, 17582, 12703, 8608, 5225, 2839, 1437, 660, 283, 122, 66, 32, 12, 8, 4, 2, 2, 2, 0, 4, 0, 2, 0, 1]
    问题描述长度：词最长1034、字最长3013、字-1数量948947、词-1数量954242、词平均长度？
    话题多少 最多16个话题绑定
    分布：[3875, 371229, 419406, 568768, 217860, 249095, 231, 118, 63, 22, 6, 12, 3, 3, 0, 0, 1]
    结论：很多少时候没有描述
    
    :param filename:
    :return:
    '''
    length = 0
    length2 = 0
    length3 = 0
    num = 0

    with open(filename,'r',encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            colum = line.split("\t")
            # print(colum)
            if colum[3] == '-1':
                words = []
                # length = length+1
            else:
                words = colum[3].split(',')
            # length += len(words)
            # length = max(length, len(words))
            length += len(words)

            if colum[5] == '-1':
                words2 = []
                # length2 = length2 + 1
            else:
                words2 = colum[5].split(',')
            # length2 = max(length2, len(words2))
            length2 += len(words2)

            if colum[6] == '-1':
                words3 = []
                # length2 = length2 + 1
            else:
                words3 = colum[6].split(',')
            length3 = max(length3, len(words3))



            # print(words)
            # print(length)
            # if(num == 0): break
            num = num+1
        # print(length/num)
        # print(length2/num)
        print(length3)


def get_question_map():
    '''
    映射questionid到标题、描述、topicid，行数1830296
    :return:
    '''
    with open("../dataset/question_info.txt", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            colum = line.split()
            info = 1
            if (colum[5] == '-1'):
                info = 0
            question_map[colum[0]] = [colum[3], info, colum[6]]

            
def get_word_map():
    '''
    映射wordid 到nparray，行数1762829
    :return:
    '''
    with open("../dataset/word_vectors_64d.txt", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            colum = line.split()
            word_map[colum[0]] = np.array([float(x) for x in colum[1:65]])

def get_topic_map():
    """
    映射topicid 到nparray，行数100000
    :return:
    """
    with open("../dataset/topic_vectors_64d.txt", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            colum = line.split()
            # print(colum)
            topic_map[colum[0]] = np.array([float(x) for x in colum[1:65]])

            
def get_member_map():
    """
    映射memberid，行数1932081
    :return:
    """
    with open("../dataset/member_info.txt", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            colum = line.split()
            member_map[colum[0]] = 1
    

def get_word_vec(str):
    '''
    把wordlist中的wordid变成wordvec的numpy的array
    :param wordlist:
    :return:
    '''
    wordlist = str.split(',')
    ret=[]
    for str in wordlist:
        ret.append(word_map[str])
    return ret


def get_topic_vec(str):
    '''
    把topiclist中的topicid变成topic的numpy的array
    :param wordlist:
    :return:
    '''
    topiclist = str.split(',')
    ret=[]
    for str in topiclist:
        ret.append(topic_map[str])
    return ret


def get_question_vec(id):
    """
    获取一个问题向量，38*64、1、16*64
    :param id: 
    :return: 
    """
    vec1 = []
    vec2 = 0
    vec3 = []
    temp = np.array([0 for i in range(0,64)])
    if question_map.__contains__(id):
        colum = question_map[id]
        # print('question', id, 'found')
    else:
        # print('question', id, 'not found')
        return 0, vec1, vec2, vec3
    # 获得38维度的标题
    if colum[0] == '-1':
        for i in range(0,38):
            vec1.append(temp)
    else:
        word_list = get_word_vec(colum[0])
        for word_vec in word_list:
            vec1.append(word_vec)

        for i in range(len(word_list),38):
            vec1.append(temp)
    # 获得有无描述
    vec2 = colum[1]
    # 获得16维度的话题
    if colum[2] == '-1':
        for i in range(0, 16):
            vec3.append(temp)
    else:
        topic_list = get_topic_vec(colum[2])
        for topic in topic_list:
            vec3.append(topic)
        for i in range(len(topic_list), 16):
            vec3.append(temp)

    return 1,vec1,vec2,vec3


def get_member_vec(id):
    '''
    判断用户是否存在，返回用户信息
    '''
    if(member_map.__contains__(id)):
        return 1
    else:
        return 0


def get_train_x(filename):
    '''
    获得所有问题的向量，存为numpy.array
    :param filename:
    :return:
    '''
    questions = []
    cnt=0
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            colum = line.split()
            f, vec1, vec2, vec3 = get_question_vec(colum[0])
            if f==0:
                continue
            else:
                cnt+=1
            vec = []
            vec.extend(vec1)
            vec.extend(vec2)
            vec.extend(vec3)
            questions.append(vec)
            # np.array(get_user_vec(colum[1]))
    print('all have',cnt,'not found!!!')
    questions_numpy = np.array(questions)
    np.save('train_x.npy', questions_numpy)
    return


def get_train_data(filename):
    '''
    生成数据，行数10166100
    '''
    question_title = []
    question_info = []
    question_topic = []
    out = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        cnt = 0
        file = 1
        for line in load_f.readlines():
            line = line.strip()
            colum = line.split()
            f, vec1, vec2, vec3 = get_question_vec(colum[0])
            if (f == 0):
                continue
            user = get_member_vec(colum[1])
            if (user == 0):
                continue
            question_title.append(vec1)
            question_info.append(vec2)
            question_topic.append(vec3)
            out.append(colum[3])
            cnt += 1
            if (cnt == 500000):
                print(cnt, file)
                #np.save('train_Qtitle_' + str(file), np.array(question_title))
                #np.save('train_Qinfo_' + str(file), np.array(question_info))
                #np.save('train_Qtopic_' + str(file), np.array(question_topic))
                #np.save('train_y_' + str(file), np.array(out))
                file += 1
                question_title = []
                question_info = []
                question_topic = []
                out = []
                cnt = 0
    if (cnt > 0):
        print(cnt, file)
        #np.save('train_Qtitle_' + str(file), np.array(question_title))
        #np.save('train_Qinfo_' + str(file), np.array(question_info))
        #np.save('train_Qtopic_' + str(file), np.array(question_topic))
        #np.save('train_y_' + str(file), np.array(out))
            

if __name__ == '__main__':
    
    # lable = getlable('../dataset/invite_info.txt')
    # np.save('train_y.npy',lable)
    # print(lable.shape)
    # analysisdata('../dataset/question_info.txt')
    # get_train_x('../dataset/invite_info.txt')
    # 清理数据库7没有出现过的question id 和 member id

    import os
    #os.system("pause")

    get_question_map()
    print('初始化questionmap成功')
    # print(question_map['Q2234111670'])
    # import os
    # os.system("pause")

    get_topic_map()
    print('初始化topicmap成功')
    #print(type(topic_map['T1']))
    #print(topic_map['T1'])

    get_word_map()
    print('初始化wordmap成功')
    #print(type(word_map['W1']))
    #print(word_map['W1'])

    get_member_map()
    print("初始化membermap成功")
    #print(type(member_map['M3807748125']))
    #print(member_map['M3807748125'])

    get_train_data('../dataset/invite_info.txt')
    
    #get_train_x('../dataset/invite_info.txt')
