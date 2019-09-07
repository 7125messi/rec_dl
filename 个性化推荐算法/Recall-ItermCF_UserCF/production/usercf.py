from __future__ import division
import sys
sys.path.append('../util')
import read
import math
import operator

def transfer_user_click(user_click):
    """
    get item by user_click
    :param user_click:{userid:[itemid1,itemid2,...]}
    :return:{itemid:[user1,user2,...]}
    """
    item_click_by_user = {}
    for user in user_click:
        item_list = user_click[user]
        for itemid in item_list:
            item_click_by_user.setdefault(itemid,[])
            item_click_by_user[itemid].append(user)
    return item_click_by_user

def base_contribution_score():
    return 1

def update_one_contribution_score(item_user_click_count):
    """
    usercf user contribution  score update v1
    :param item_user_click_count:how many user clicked this item
    :return:contribution score
    """
    return 1/math.log10(1+item_user_click_count)

def update_two_contribution_score(click_time_one,click_time_two):
    """
    usercf user contributionscore update v2
    :param click_time_one:different user action time to the same item
    :param click_time_two:different user action time to the same item
    :return:contribution score
    """
    delta_time = abs(click_time_one - click_time_two)
    total_sec = 60 * 60 * 24
    delta_time = delta_time / total_sec
    return 1 / (1 + delta_time)

def cal_user_sim(item_click_by_user,user_click_time):
    """
    get user sim info
    :param item_click_by_user:{itemid:[itemid1,itemid2,...]}
    :return:{itemid_i,{itemid_j:sim_score}}
    """
    co_appear = {}
    user_click_count = {}
    for itemid,user_list in item_click_by_user.items():
        for index_i in range(0,len(user_list)):
            user_i = user_list[index_i]
            user_click_count.setdefault(user_i,0)
            user_click_count[user_i] += 1
            if user_i + "_" +itemid not in user_click_time:
                click_time_one = 0
            else:
                click_time_one = user_click_time[user_i + "_" + itemid]
            for index_j in range(index_i+1,len(user_list)):
                user_j = user_list[index_j]
                if user_j + "_" + itemid not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user_j + "_" + itemid]
                co_appear.setdefault(user_i,{})
                co_appear[user_i].setdefault(user_j,0)
                # co_appear[user_i][user_j] += base_contribution_score()
                # co_appear[user_i][user_j] += update_one_contribution_score(len(user_list))
                co_appear[user_i][user_j] += update_two_contribution_score(click_time_one,click_time_two)


                co_appear.setdefault(user_j, {})
                co_appear[user_j].setdefault(user_i, 0)
                # co_appear[user_j][user_i] += base_contribution_score()
                # co_appear[user_j][user_i] += update_one_contribution_score(len(user_list))
                co_appear[user_j][user_i] += update_two_contribution_score(click_time_one,click_time_two)



    user_sim_info = {}
    user_sim_info_sorted = {}
    for user_i,relate_user in co_appear.items():
        user_sim_info.setdefault(user_i,{})
        for user_j,cotime in relate_user.items():
            user_sim_info[user_i].setdefault(user_j,0)
            user_sim_info[user_i][user_j] = cotime/math.sqrt(user_click_count[user_i]*user_click_count[user_j])

    for user in user_sim_info:
        user_sim_info_sorted[user] = sorted(user_sim_info[user].items(),key=operator.itemgetter(1),reverse=True)
    return user_sim_info_sorted


def cal_recom_result(user_click,user_sim):
    """
    recom by  usercf
    :param user_click:{userid:[itemid1,itemid2,...]}
    :param user_sim:{userid:[(userid1,score1),(userid2,score2),...]}
    :return:{userid:{itemid:recom_score}}
    """
    recom_result = {}
    topk_user = 3
    item_num = 5
    for user,item_list in user_click.items():
        tmp_dict = {}
        for itemid in item_list:
            tmp_dict.setdefault(itemid,1)
        recom_result.setdefault(user,{})
        for zuhe in user_sim[user][:topk_user]:
            userid_j,sim_score = zuhe
            if userid_j not in user_click:
                continue
            for itemid_j in user_click[userid_j][:item_num]:
                recom_result[user].setdefault(itemid_j,sim_score)
    return recom_result

def debug_user_sim(user_sim):
    """
    print user sim result
    :param user_sim:{userid:[(userid1,score1),(userid2,score2),...]}
    """
    topk=5
    fix_user = '1'
    if fix_user not in user_sim:
        print("Invalid user")
        return
    for zuhe in user_sim[fix_user][:topk]:
        userid,score = zuhe
        print(fix_user + "\tsim_user_" + userid + "\t" + str(score))

def debug_recom_result(item_info,recom_result):
    """
    print recom result for a fixed user
    :param item_info:{itemid:[title,genres]}
    :param recom_result:{userid:{itemid:recom_score}}
    """
    fix_user = "1"
    if fix_user not in recom_result:
        print("Invalid user for recoming result")
        return
    for itemid in recom_result["1"]:
        if itemid not in item_info:
            continue
        recom_score = recom_result["1"][itemid]
        print("recom result:" + ",".join(item_info[itemid]) + "\t" + str(recom_score))


def main_flow():
    """
    计算user 相似度矩阵
    推荐物品
    :return:
    """
    # user_click,user_click_time = read.get_user_list('../data/ratings.txt')
    # item_click_by_user = transfer_user_click(user_click)
    # user_sim = cal_user_sim(item_click_by_user,user_click_time)
    # recom_result = cal_recom_result(user_click,user_sim)
    # print(recom_result["1"])

    user_click, user_click_time = read.get_user_list('../data/ratings.txt')
    item_info = read.get_item_info('../data/movies.txt')
    item_click_by_user = transfer_user_click(user_click)
    user_sim = cal_user_sim(item_click_by_user,user_click_time)
    # debug_user_sim(user_sim)
    recom_result = cal_recom_result(user_click, user_sim)
    debug_recom_result(item_info,recom_result)


if __name__ == "__main__":
    main_flow()
