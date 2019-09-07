from __future__ import division
import sys
import os
sys.path.append('../util')
import read
import math
import operator
from tqdm import tqdm

def base_contribute_score():
    """
    item cf base sim contribution score
    """
    return 1

def update_one_contribute_score(user_total_click_num):
    """
    :param user_total_click_num:item cf update sim contribution score by user
    """
    return 1/math.log10(1+user_total_click_num)

def update_two_contribute_score(click_time_one,click_time_two):
    """
    item cf update two sim contribution score by user
    """
    delta_time = abs(click_time_one-click_time_two)
    total_sec = 60*60*24
    delta_time = delta_time/total_sec
    return 1/(1+delta_time)

def cal_item_sim(user_click,user_click_time):
    """
    :param user_click:k userid:v [itemid1,itemid2...]
    :param user_click_time同一用户对不同item点击时间差，时间差越小，对最终相似贡献越大
    :return:k itemid_i,v ——> k itemid_j,v simscore
    """
    co_appear = {}
    item_user_click_time = {}
    for user,itemlist in tqdm(user_click.items()):
        for index_i in range(0,len(itemlist)):
            itemid_i = itemlist[index_i]
            item_user_click_time.setdefault(itemid_i,0)
            item_user_click_time[itemid_i] += 1
            for index_j in range(index_i+1,len(itemlist)):
                itemid_j = itemlist[index_j]
                if user + "_" + itemid_i not in user_click_time:
                    click_time_one = 0
                else:
                    click_time_one = user_click_time[user + "_" + itemid_i]
                if user + "_" + itemid_j not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user + "_" + itemid_j]
                co_appear.setdefault(itemid_i,{})
                co_appear[itemid_i].setdefault(itemid_j,0)
                # co_appear[itemid_i][itemid_j] += base_contribute_score()
                # co_appear[itemid_i][itemid_j] += update_one_contribute_score(len(itemlist))
                co_appear[itemid_i][itemid_j] += update_two_contribute_score(click_time_one,click_time_two)

                co_appear.setdefault(itemid_j,{})
                co_appear[itemid_j].setdefault(itemid_i,0)
                # co_appear[itemid_j][itemid_i] += base_contribute_score()
                # co_appear[itemid_j][itemid_i] += update_one_contribute_score(len(itemlist))
                co_appear[itemid_j][itemid_i] += update_two_contribute_score(click_time_one,click_time_two)

    item_sim_score = {}
    item_sim_score_sorted = {}
    for itemid_i,relate_item in tqdm(co_appear.items()):
        for itemid_j,co_time in relate_item.items():
            sim_score = co_time/math.sqrt(item_user_click_time[itemid_i]*item_user_click_time[itemid_j])
            item_sim_score.setdefault(itemid_i,{})
            item_sim_score[itemid_i].setdefault(itemid_j,0)
            item_sim_score[itemid_i][itemid_j] = sim_score
    for itemid in tqdm(item_sim_score):
        # 这里sorted排序的话，这里的value将不再是字典，而变成一个list,其元素为元祖（itemid,score）
        item_sim_score_sorted[itemid] = sorted(item_sim_score[itemid].items(),key=operator.itemgetter(1),reverse=True)
    return item_sim_score_sorted

def cal_recom_result(sim_info,user_click):
    """
    :param sim_info:item sim dict
    :param user_click: user click dict
    :return: a dict.key userid, value dict(value_key itemid,value_value recom_score)
    recom by itemcf
    """
    recent_click_num = 3
    topk = 5
    recom_info = {}
    for user in tqdm(user_click):
        click_list = user_click[user]
        recom_info.setdefault(user,{})
        for itemid in click_list[:recent_click_num]:
            if itemid not in sim_info:
                continue
            for itemsimzuhe in sim_info[itemid][:topk]:
                itemsimid = itemsimzuhe[0]
                itemsimscore = itemsimzuhe[1]
                recom_info[user][itemsimid] = itemsimscore
    return recom_info

def degbug_itemsim(item_info,sim_info):
    """
    :param item_info:{itemid:[title,genres]}
    :param sim_info:{itemid:[(itemid1,simscore),(itemid2,simscore)]}
    """
    fixed_itemid = "1"
    if fixed_itemid not in item_info:
        print("Invalid Itemid")
        return
    [title_fix,genres_fix] = item_info[fixed_itemid]
    for zuhe in sim_info[fixed_itemid][:5]:
        itemid_sim = zuhe[0]
        sim_score = zuhe[1]
        if itemid_sim not in item_info:
            continue
        [title,genres] = item_info[itemid_sim]
    print(title_fix + "\t" + genres_fix + "\tsim:" + title + "\t" + genres + "\t" + str(sim_score))

def debug_recomresult(recom_result,item_info):
    """
    :param recom_result:{userid:{itemid:score}}
    :param item_info:{itemid:[title,genres]}
    :return:
    """
    user_id = "1"
    if user_id not in recom_result:
        print("Invalid Result")
        return
    for zuhe in sorted(recom_result[user_id].items(),key=operator.itemgetter(1),reverse=True):
        itemid,score = zuhe
        if itemid not in item_info:
            continue
        print(",".join(item_info[itemid]) + "\t" + str(score))

def main_flow():
    """
    main flow of itemcf
    (1) 计算得到item的相似度
    (2) 根据item的相似度来进行推荐
    """
    # user_click,user_click_time = read.get_user_list('../data/ratings.txt')
    # item_info = read.get_item_info('../data/movies.txt')
    # sim_info = cal_item_sim(user_click,user_click_time)
    # recom_result = cal_recom_result(sim_info,user_click)
    # print(recom_result['1'])

    user_click,user_click_time = read.get_user_list('../data/ratings.txt')
    item_info = read.get_item_info('../data/movies.txt')
    sim_info = cal_item_sim(user_click,user_click_time)
    degbug_itemsim(item_info,sim_info)
    recom_result = cal_recom_result(sim_info,user_click)
    debug_recomresult(recom_result, item_info)

if __name__ == "__main__":
    main_flow()
