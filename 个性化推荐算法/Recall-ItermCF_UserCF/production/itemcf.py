import sys
import os
sys.path.append('../util')
import read

def main_flow():
    """
    main flow of itemcf
    (1) 计算得到item的相似度
    (2) 根据item的相似度来进行推荐
    """
    user_click = read.get_user_click('../data/ratings.txt')
    sim_info = cal_item_sim(user_click)
    recom_result = cal_recom_result(sim_info,user_click)

def cal_item_sim(user_click):
    """
    :param user_click:k userid:v [itemid1,itemid2...]
    :return:k itemid_i,v ——> k itemid_j,v simscore
    """
    pass
