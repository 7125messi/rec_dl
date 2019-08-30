import os
def get_item_info(input_file):
    """
    get item info:[title,genre]
    Args:
        input_file:item info file
    Return:
    a dict: key itemid ,value:[title,genre]
    """
    if not os.path.exists(input_file): #文件不存在返回空字典
        return {}
    item_info = {}    #存放item信息
    linenum = 0       #标记第几行为了过滤第一行无用信息
    fp = open(input_file,'rb')   #打开文件
    for line in fp :
        if linenum == 0 :  #如果是第一行linenum加1 继续
            linenum += 1
            continue
        line = line.decode()    #将str型转为byter型
        item = line.strip().split(',')  #将一行信息按‘，’分隔成三部分信息
        if len(item) <3 : #如果信息数少于3 直接过滤掉
            continue
        elif len(item) == 3 :  #如果信息数等于3 直接赋值
            itemid,title,genre = item[0],item[1],item[2]
        elif len(item) > 3 :   #如果信息数大于3 第一部分为id，最后一部分是分类，当中是title
            itemid = item[0]
            genre = item[-1]
            title = ','.join(item[1:-1])
        item_info[itemid] = [title,genre]  #将信息加入字典
    fp.close() #关闭文件
    return item_info


def get_ave_score(input_file) : #得到item的平均评分
    """
    get item ave rating score
    Args: user rating file
    Return: a dict
       key itemid,value ave_score
    """
    if not os.path.exists(input_file) :
        return {}
    linenum = 0
    record_dict = {}
    score_dict = {}
    fp = open(input_file)
    for line in fp :
        if linenum == 0 :
            linenum += 1
            continue
        # line = line.decode()  
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        itemid,rating = item[1],item[2]
        if itemid not in record_dict :
            record_dict[itemid] = [0,0]
        record_dict[itemid][0] += 1
        record_dict[itemid][1] += float(rating)
    fp.close()
    for itemid in record_dict :
        score_dict[itemid] = round(record_dict[itemid][1] / record_dict[itemid][0],3) #round(,3)保留3位有效数字
    return score_dict
    
def get_train_data(input_file):
    """
    get train data for LFM model train
    Args:
        input_file: user item ……]
    """
    if not os.path.exists(input_file):
        return []
    score_th = 4.0
    score_dict = get_ave_score(input_file)
    neg_dict = {}
    pos_dict = {} #存储正负样本
    train_dict = []  #训练样本
    linenum = 0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',') #strip()函数为删除空白符
        if len(item) < 4:
            continue
        userid,itemid,rating = item[0],item[1],float(item[2])
        if userid not in pos_dict :
            pos_dict[userid] = []
        if userid not  in neg_dict :
            neg_dict[userid] = []
        if rating >= score_th :
            pos_dict[userid].append((itemid,1))
        else:
            score = score_dict.get(itemid,0) #dict.get(key,default=None) key是要查找的关键字，default是如果指定键的值不存在时，返回的默认值
            neg_dict[userid].append((itemid,score)) #由于要做负样本采样，所以负样本存储的是用户平均的评分
    fp.close()
    for userid in pos_dict : #均衡正负样本的采样
        date_num = min(len(pos_dict[userid]),len(neg_dict[userid])) #获取训练样本的数目 
                                                                    #真实情况下正样本数目远小于负样本数目，但此处为测试样本，所以需要做一次比较
        if date_num > 0 : # 如果正负样本数目大于0 ，先将正样本存入到训练样本中
            train_dict += [(userid,zuhe[0],zuhe[1]) for zuhe in pos_dict[userid]][:date_num]
        else :
            continue  #正样本数目为0 跳过
        # sorted_neg_list = sorted(neg_dict[userid],lambda element:element[1],reverse = True)[:date_num] #对负样本进行排序且取前data_num组数据作为负样本
        sorted_neg_list = sorted(neg_dict[userid],key = lambda element:element[1],reverse = True)[:date_num]  #在python3 中需要加上key这个关键字
        train_dict += [(userid,zuhe[0],0) for zuhe in sorted_neg_list]
    return train_dict


if __name__ == "__main__":
    item_dict = get_item_info("../data/movies.txt")
    print(len(item_dict))
    print(item_dict["11"])
    ave_score = get_ave_score("../data/ratings.txt")
    print (len(ave_score))
    print(ave_score["31"])
    train_data = get_train_data("../data/ratings.txt")
    print(len(train_data))