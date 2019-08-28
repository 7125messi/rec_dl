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

def get_graph_from_data(input_file):
    if not os.path.exists(input_file):
        return {}
    graph = {}
    linenum = 0
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(",")
        if len(item) < 3:
            continue
        userid,itemid,rating = item[0],"item_" + item[1],item[2]
        if float(rating) < score_thr:
            continue
        if userid not in graph:
            graph[userid] = {}
        graph[userid][itemid] = 1
        if itemid not in graph:
            graph[itemid] = {}
        graph[itemid][userid] = 1
    fp.close()
    return graph

if __name__ == "__main__":
    graph = get_graph_from_data('../data/ratings.txt')
    print(graph["38"])