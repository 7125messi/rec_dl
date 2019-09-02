from __future__ import absolute_import
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
import read

# matrix util for personal rank 矩阵化形式
# 所有顶点PR值排序，进而得到所有用户的推荐结果

# 根据用户二分图得到m稀疏矩阵
def graph_to_m(graph):
    """
    args:
        graph:user item graph
    return:
        a sparse matrix M
        a coo_matrix,sparse matrix M
        a list,total user item point
        a dict,map all the point to row index
    """
    vertex = list(graph.keys()) # 先定义所有顶点
    address_dict = {}
    total_len = len(vertex)
    for index in range(len(vertex)):
        address_dict[vertex[index]] = index # 每一行对应哪个顶点
    row = [] # 行索引
    col = [] # 列索引
    data = [] # 对应数值
    for element_i in graph:
        weight = round(1/len(graph[element_i]),3)
        row_index = address_dict[element_i]
        for element_j in graph[element_i]:
            col_index = address_dict[element_j]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row = np.array(row)
    col = np.array(col)
    weight = np.array(data)
    m = coo_matrix((data,(row,col)),shape=(total_len,total_len))
    return m,vertex,address_dict

def mat_all_point(m_mat,vertex,alpha):
    """
    get E-alpha*m_mat.T
    args:
        m_mat:
        vertex:total item and user point
        alpha:the prob for random walking随机游走概率
    return:
        sparse matrix
    """
    # np.eys()初始化单位矩阵易超内容
    total_len = len(vertex)
    row = []
    col = []
    data = []
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    eye_t = coo_matrix((data,(row,col)),shape=(total_len,total_len))
    print(eye_t.shape)
    return eye_t.tocsr() - alpha*m_mat.tocsr().transpose()


if __name__ == "__main__":
    graph = read.get_graph_from_data('../data/ratings.txt')
    m,vertex,address_dict = graph_to_m(graph)
    print(len(address_dict))
    print(m.todense())
    print(m.todense().shape)

    res_m = mat_all_point(m,vertex,0.8)
    print(res_m.shape)
    print(res_m)