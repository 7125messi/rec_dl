https://github.com/dav/word2vec
# 注意
- src/*.c  请注释掉#include <malloc.h>
- 下载，切换到word2vec目录下，进入./src文件夹，然后输入make，回车，之后会在bin文件下出现很多可执行文件
```shell
$ ./bin/word2vec
WORD VECTOR estimation toolkit v 0.1c

Options:
Parameters for training:
        -train <file>
                Use text data from <file> to train the model
        -output <file>
                Use <file> to save the resulting word vectors / word clusters
        -size <int>
                Set size of word vectors; default is 100
        -window <int>
                Set max skip length between words; default is 5
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the training data
                will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -hs <int>
                Use Hierarchical Softmax; default is 0 (not used)
        -negative <int>
                Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        -threads <int>
                Use <int> threads (default 12)
        -iter <int>
                Run more training iterations (default 5)
        -min-count <int>
                This will discard words that appear less than <int> times; default is 5
        -alpha <float>
                Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
        -classes <int>
                Output word classes rather than word vectors; default number of classes is 0 (vectors are written)
        -debug <int>
                Set the debug mode (default = 2 = more info during training)
        -binary <int>
                Save the resulting vectors in binary moded; default is 0 (off)
        -save-vocab <file>
                The vocabulary will be saved to <file>
        -read-vocab <file>
                The vocabulary will be read from <file>, not constructed from the training data
        -cbow <int>
                Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)

Examples:
./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
```

```shell
sh train.sh
```
```shell
# 动态查看item=1找出最相近的 binary = 1 二进制
./bin/distance ../data/item_vec.txt

(nlp) [zhaoyadong@15:53:51]~/opt/git/rec_dl/Recall-Item2vec/production$ ../bin/distance ../data/item_vec.txt 
Enter word or sentence (EXIT to break): 1

Word: 1  Position in vocabulary: 17

                                              Word       Cosine distance
------------------------------------------------------------------------
                                              3624              0.295818
                                              2581              0.276785
                                             99114              0.276024
                                              8972              0.267305
                                              1081              0.267098
                                              3917              0.265685
                                             31410              0.263051
                                             33660              0.261018
                                              8207              0.259878
                                             60040              0.259552
                                              2628              0.259085
                                                39              0.258058
                                             54190              0.254846
                                              2997              0.254223
                                            112290              0.249965
                                            177593              0.249573
                                             71033              0.247637
                                              5816              0.247097
                                             54001              0.242311
                                              1231              0.239053
                                              2359              0.238679
                                             53996              0.227248
                                              3334              0.226643
                                              2009              0.226416
                                            157296              0.225725
                                              8961              0.225184
                                              8368              0.224874
                                               101              0.224004
                                             27611              0.220415
                                              1061              0.219882
                                              3317              0.217398
                                              2117              0.213797
                                                28              0.213689
                                              7090              0.211147
                                              1717              0.210067
                                               750              0.209975
                                             57368              0.209532
                                             95510              0.209091
                                              3159              0.208904
                                              6440              0.208476
Enter word 
```

# 根据item embedding向量得到相似度矩阵
```shell
sh train.sh ../data/train_data.txt
```

# 如何训练word2vec——>item embedding ——> item sim matrix
