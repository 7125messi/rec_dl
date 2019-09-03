import numpy as np
import sys
sys.path.append("../util")
import read as read

def lfm_train(train_data,F,alpha,beta,step) :
    """
    Args:
        train_data: train_data for lfm
        F:user vector len , item vector len
        alpha:regularization factor
        beta: lrarning rate
        step: iteration num
    Return:
        dict: key userid,value vector:np.ndarray
        dict: key itemid,value vector:np.ndarray
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        for data_instance in train_data:
            userid,itemid,label = data_instance
            if userid not in user_vec:  #第一次训练时将其加入到user_vec中
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:  #同理
                item_vec[itemid] = init_model(F)
            delta = label - model_predict(user_vec[userid],item_vec[itemid])
            for index in range(F) : #更新user_vec和item_vec的向量
                user_vec[userid][index] += beta * (delta * item_vec[itemid][index] - alpha * user_vec[userid][index])
                item_vec[itemid][index] += beta * (delta * user_vec[userid][index] - alpha * item_vec[itemid][index])
            beta = beta * 0.9
        # print(step_index)
        # print (user_vec[userid])
        # print(item_vec[itemid])
        # print(model_predict(user_vec[userid],item_vec[itemid]))
    return  user_vec,item_vec

def init_model(vactor_len):
    """
    Agrs:
       the len of vactor
    Return:
        a ndarray 
    """
    vector = np.random.randn(vactor_len)
    return vector

def model_predict(use_vec,item_vec) :
    """
    Agrs:
        user and item vector
    Return:
        a num
    """
    res = np.dot(use_vec,item_vec)/(np.linalg.norm(use_vec) * np.linalg.norm(item_vec)) #np.dot() 求向量的内积 np.linalg.norm()求向量的模
    return res

def model_train_process():
    """
    test lfm model train
    """
    train_data = read.get_train_data("../data/ratings.txt")
    user_vec,item_vec = lfm_train(train_data,10,0.01,0.1,5)

    # 全部用户推荐
    # for userid in user_vec:
    #     recom_list = give_recom_result(user_vec,item_vec,userid)
    #     ana_recom_result(train_data,userid,recom_list)

    # 某一用户推荐
    recom_list = give_recom_result(user_vec,item_vec,"38")
    ana_recom_result(train_data,"38",recom_list)

def give_recom_result(user_vec,item_vec,userid):
    """
    use lfm model result giv fix userid recommend result
    Args:
        user_vec:lfm model result
        item_vec:lfm model result
        userid: fix userid
    """
    fix_num = 10
    if userid not in user_vec:
        return []
    record = {}
    recom_list = []
    user_vector = user_vec[userid]
    for itemid in item_vec :
        itme_vector = item_vec[itemid]
        res = np.dot(user_vector,itme_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(itme_vector))
        record[itemid] = res
    for zuhe in sorted(record.items(),key = lambda item:item[1],reverse = True)[:fix_num] :
        itemid = zuhe[0]
        score = round(zuhe[1],3)
        recom_list.append((itemid,score))
    return recom_list
    
def ana_recom_result(train_data,userid,recom_list):
    """
    debug recom result for userid
    Args:
        train_data: train data for model
        userif : fix user
        recome_list: lfm makes recommend list for userid
    """
    item_info = read.get_item_info("../data/movies.txt")
    for data_instance in train_data:
        tp_userid,itemid,label = data_instance
        if userid == tp_userid and label == 1:
            print(itemid,item_info[itemid])
    print("recommend result")
    for zuhe in recom_list:
        print(zuhe[0],item_info[zuhe[0]])

if __name__ == "__main__":
    model_train_process()
