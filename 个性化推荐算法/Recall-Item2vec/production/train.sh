#!/usr/bin/env bash

train_file = $1
item_vec_file = $2

# -binary 1 保存结果二进制,观察训练结果是否合理
# ../bin/word2vec -train $train_file -output $item_vec_file -size 100 -window 3 -sample 1e-4 -negative 5 -hs 0 -binary 1 -cbow 0 -iter 10
# ../bin/word2vec -train ../data/train_data.txt -output ../data/item_vec_bin.txt -size 100 -window 3 -sample 1e-4 -negative 5 -hs 0 -binary 1 -cbow 0 -iter 10





# -binary 0 不保存结果二进制
# ../bin/word2vec -train ../data/train_data.txt -output ../data/item_vec.txt -size 100 -window 3 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 10

../bin/word2vec -train $train_file -output $item_vec_file -size 100 -window 3 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 10
