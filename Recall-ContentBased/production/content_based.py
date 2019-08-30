from __future__ import division
import os
import numpy
import operator
import sys
sys.path.append('../util')
import read

# 用户刻画和线上推荐
def get_up(item_cate,input_file):
    """
    :param item_cate:key itemid,value:dict,key category,value ratio
    :param input_file: user rating file
    :return: a dict:key userid,value[(category,ratio),(category1,ratio1 ),...]
    """
    if not os.path.exists(input_file):
        return {}
    record = {}
    up = {}
    topk = 2
    linenum = 0
    score_thr = 4.0
    with open(input_file,encoding='utf-8') as fp:
        for line in fp:
            if linenum == 0:
                linenum += 1
                continue
            item = line.strip().split(',')
            if len(item) < 4:
                continue
            userid,itemid,rating,timestamp = item[0],item[1],float(item[2]),item[3]
            if rating < score_thr:
                continue
            # 无法获得类别也要过滤
            if itemid not in item_cate:
                continue
            time_score = get_time_score(timestamp)
            if userid not in record:
                record[userid] = {}
            for fix_cate in item_cate[itemid]:
                if fix_cate not in record[userid]:
                    record[userid][fix_cate] = 0
                record[userid][fix_cate] += rating * time_score * item_cate[itemid][fix_cate]

    # 对每个用户进行排序
    for userid in record:
        if userid not in up:
            up[userid] = []
            total_score = 0
        for zuhe in sorted(record[userid].items(),key=operator.itemgetter(1),reverse=True)[:topk]:
            up[userid].append([zuhe[0],zuhe[1]])
            # up[userid].append((zuhe[0], zuhe[1]))
            total_score += zuhe[1]
        for index in range(len(up[userid])):
            # up[userid][index][1] = (up[userid][index][0],round(up[userid][index][1] / total_score, 3))
            up[userid][index][1] = round(up[userid][index][1]/total_score,3)
    return up

def get_time_score(timestamp):
    """
    :param timestamp:input timestamp
    :return: time score
    """
    # max timestamp:1493846415
    fix_time_stamp = 1493846415
    total_sec = 24*60*60
    delta = (fix_time_stamp - int(timestamp))/total_sec
    # delta = (fix_time_stamp - int(timestamp)) / total_sec/100
    # print(delta)
    return round(1/1+delta,3)

# 推荐函数
# 基于内容的推荐
# 用户刻画up存入kv，item倒排cate_item_sort存入搜索引擎，推荐的过程是在线实时，分别请求Kv 和搜索引擎获取倒排序
def recom(cate_item_sort,up,userid,topk=10):
    """
    :param cate_item_sort:reverse sort
    :param up: user profile
    :param userid: userid to recom
    :param topk: recom num
    :return:userid:[itemid1,itemid2]
    """
    if userid not in up:
        return {}
    recom_result = {}
    if userid not in recom_result:
        recom_result[userid] = []
    for zuhe in up[userid]:
        cate = zuhe[0]
        ratio = zuhe[1]
        num = int(topk*ratio) + 1
        if cate not in cate_item_sort:
            continue
        recom_list = cate_item_sort[cate][:num]
        recom_result[userid] += recom_list
    return recom_result

def run_main():
    ave_score = read.get_ave_score('../data/ratings.txt')
    item_cate,cate_item_sort = read.get_item_cate(ave_score,'../data/movies.txt')
    print(cate_item_sort)
    up = get_up(item_cate,'../data/ratings.txt')
    print(len(up))
    print(up['38'])
    print(recom(cate_item_sort,up,38))

if __name__ == "__main__":
    run_main()