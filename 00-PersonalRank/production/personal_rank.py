from __future__ import absolute_import
from __future__ import division
import operator
import sys
sys.path.append('../util')
import read
import mat_util
from scipy.sparse.linalg import gmres   # 解Ax=B稀疏矩阵
import numpy as np

def personal_rank(graph,root,alpha,iter_num,recom_num=10):
    """
    Args:
        graph:user item graph
        root:the fixed user for which to recom
        alpha:the prob to go to random walk,1-alpha回到起点
        iter_num:iteration num
        recom_num:recom item num
    Return:
        a dict key itemid,value pr
    """
    rank = {} # 存储所有顶点的pr值
    rank = {point:0 for point in graph} # 除root顶点，其余所有顶点初始化为0 
    rank[root] = 1 # root顶点初始化为1
    recom_result = {} # 定义输出数据结构
    for iter_index in range(iter_num):
        tmp_rank = {}
        tmp_rank = {point:0 for point in graph}
        for out_point,out_dict in graph.items():
            for inner_point,value in graph[out_point].items():
                tmp_rank[inner_point] += round(alpha*rank[out_point]/len(out_dict),4)
                if inner_point == root:
                    tmp_rank[inner_point] += round(1-alpha,4)
        if tmp_rank == rank:
            print("out" + str(iter_index))
            break
        rank = tmp_rank
    right_num = 0
    for zuhe in sorted(rank.items(),key=operator.itemgetter(1),reverse=True):
        point,pr_score = zuhe[0],zuhe[1]
        if len(point.split("_")) < 2:
            continue
        if point in graph[root]:
            continue
        recom_result[point] = pr_score
        right_num += 1
        if right_num > recom_num:
            break
    return recom_result

def personal_rank_mat(graph,root,alpha,recom_num = 10):
    """
    Args:
        graph:user item graph
        root:the fixed user for which to recom
        alpha:the prob to go to random walk,1-alpha回到起点
        recom_num:recom item num
    Return:
        a dict key itemid,value pr score
    A*r = r0 n*1矩阵
    """
    m,vertex,address_dict = mat_util.graph_to_m(graph)
    if root not in address_dict:
        return {}
    score_dict = {}
    recom_dict = {}
    mat_all = mat_util.mat_all_point(m,vertex,alpha)
    index = address_dict[root]
    initial_list = [[0] for row in range(len(vertex))]
    initial_list[index] = [1]
    r_zero = np.array(initial_list)
    res = gmres(mat_all,r_zero,tol=1e-8)[0]
    for index in range(len(res)):
        point = vertex[index]
        if len(point.strip().split("_")) < 2:
            continue
        if point in graph[root]:
            continue
        score_dict[point] = round(res[index],3)
    for zuhe in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True):
        point,score = zuhe[0],zuhe[1]
        recom_dict[point] = score
    return recom_dict

def get_one_user_recom():
    user = '38'
    alpha = 0.8  # 0.7
    graph = read.get_graph_from_data('../data/ratings.txt')
    iter_num = 100 # 15
    recom_result = personal_rank(graph,user,alpha,iter_num)
    [print(i) for i in recom_result]

    print("========================================")
    # 解析，便于理解
    item_info = read.get_item_info('../data/movies.txt')
    # 将用户现在感兴趣的item打印出来
    for itemid in graph[user]:
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])
    print("=================推荐结果=================")
    # 将用户可能感兴趣的结果打印出来
    for itemid in recom_result:
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])
        print(recom_result[itemid])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

def get_one_user_by_mat():
    user = '38'
    alpha = 0.8
    graph = read.get_graph_from_data('../data/ratings.txt')
    recom_result = personal_rank_mat(graph,user,alpha,10)
    return recom_result



if __name__ == "__main__":
    # recom_result_base = get_one_user_recom()
    # recom_result_mat = get_one_user_by_mat()
    # num = 0
    # for ele in recom_result_base:
    #     if ele in recom_result_mat:
    #         num += 1
    # print(num)

    # get_one_user_recom()
    get_one_user_by_mat()