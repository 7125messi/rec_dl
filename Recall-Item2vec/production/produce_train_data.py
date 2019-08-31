#-*-coding: utf-8 -*-
import os
import sys

# produce training data(item sequence) for item2vec
def produce_training_data(input_file,out_file):
    """
    args:
        input_file:user behavior file
        out_file:output file
    """
    if not os.path.exists(input_file):
        return
    record = {} # 记录用户喜欢过的物品
    linenum = 0
    score_thr = 4
    with open(input_file) as fp:
        for line in fp:
            if linenum == 0:
                linenum += 1
                continue
            item = line.strip().split(",")
            # item2vec忽略用户时间信息
            if len(item) < 4:
                continue
            userid,itemid,rating = item[0],item[1],float(item[2])
            if rating < score_thr:
                continue
            if userid not in record:
                record[userid] = []
            record[userid].append((itemid))
    with open(out_file,'w+') as fw:
        for userid in record:
            fw.write(" ".join(record[userid]) + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage:python xx.py inputfile outfile")
        sys.exit()
    else:
        inputfile = sys.argv[1]
        outputfile  = sys.argv[2]
        produce_training_data(inputfile,outputfile)

    # produce_training_data('../data/ratings.txt','../data/train_data.txt')
