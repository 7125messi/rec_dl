import os
import operator
import numpy as np
import sys

# numpy非科学技术法显示
np.set_printoptions(suppress=True)


# produce item sim file
def load_item_vec(input_file):
    """
    :param input_file: item vec file
    :return: dict:key is itemid,value is [num1,num2,...]
    """
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    item_vec = {}
    with open(input_file) as fp:
        for line in fp:
            if linenum == 0:
                linenum += 1
                continue
            item = line.strip().split(' ')
            if len(item) < 101:
                continue
            itemid = item[0]
            if itemid == '</s>':
                continue
            item_vec[itemid] = np.array([float(eye) for eye in item[1:]])
    return item_vec

def cal_item_sim(item_vec,itemid,output_file):
    """
    :param item_vec: item embedding vector
    :param itemid: fixed itemid to calc item sim
    :param output_file: the file to store result
    :return:None
    """
    if itemid not in item_vec:
        return
    score = {}
    fix_item_vec = item_vec[itemid]
    for tmp_itemid in item_vec:
        if tmp_itemid == itemid:
            continue
        tmp_itemvec = item_vec[tmp_itemid]
        fenmu = np.linalg.norm(fix_item_vec) * np.linalg.norm(tmp_itemvec)
        if fenmu == 0:
            score[tmp_itemid] = 0
        else:
            score[tmp_itemid] = round(np.dot(fix_item_vec,tmp_itemvec)/fenmu,3)
    with open(output_file,'w+') as fw:
        out_str = itemid + "\t"
        tmp_list = []
        for zuhe in sorted(score.items(),key=operator.itemgetter(1),reverse=True):
            # tmp_list.append(zuhe[0] + "_" + str(zuhe[1])) # 调试
            tmp_list.append(zuhe[0])
        out_str += ";".join(tmp_list)
        fw.write(out_str + '\n')

def run_main(input_file,output_file):
    item_vec = load_item_vec(input_file)
    cal_item_sim(item_vec,"318",output_file)


if __name__ == "__main__":
    # item_vec  = load_item_vec('../data/item_vec.txt')
    # print(len(item_vec))
    # print(item_vec["318"])

    # run_main('../data/item_vec.txt','../data/sim_result.txt')

    if len(sys.argv) < 3:
        print("usage:python xx.py inputfile outfile")
        sys.exit()
    else:
        inputfile = sys.argv[1]
        outputfile  = sys.argv[2]
        run_main(inputfile,outputfile)

