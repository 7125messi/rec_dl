import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import umap
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import time
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)

def data_process(input_file_user,input_file_item):
    if not os.path.exists(input_file_user):
        return
    if not os.path.exists(input_file_item):
        return
    df_user = pd.read_csv(input_file_user,sep=',')
    df_item = pd.read_csv(input_file_item,sep=',')
    df = pd.merge(df_user,df_item,how='left',on='movieId')
    # print(df.head())
    # print(len(df))
    # 准备数据
    df['movieId'] = df['movieId'].astype(str)
    customers = df["userId"].unique().tolist()

    # print(len(customers))
    # 数据集中有610个消费者,对于这些消费者，我们将提取他们的购买历史。换句话说，我们可以有610个购买序列

    # 打乱消费者id
    random.shuffle(customers)
    # 提取90%的消费者
    customers_train = [customers[i] for i in range(round(0.9*len(customers)))]
    # print(customers_train)
    # 切分为训练集和验证集
    train_df = df[df['userId'].isin(customers_train)]
    validation_df = df[~df['userId'].isin(customers_train)]

    # 为后续推荐做准备
    # 创建一个商品id和商品描述的字典，将商品的描述映射到其id
    products = train_df[["movieId", "title","genres","rating","timestamp"]]
    # 去重
    products.drop_duplicates(inplace=True, subset='movieId', keep="last")
    # 创建一个商品id和商品描述的字典
    products_dict = products.groupby('movieId')[["title","genres","rating","timestamp"]].apply(list).to_dict()
    print(products_dict['1']) # 字典测试

    # 数据集中为训练集和验证集创建消费者购买的序列
    # 存储消费者的购买历史
    purchases_train = []
    # 用商品代码填充列表
    for i in tqdm(customers_train):
        temp = train_df[train_df["userId"] == i]["movieId"].tolist()
        purchases_train.append(temp)
    # print(purchases_train)
    # print(len(purchases_train))
    purchases_val = []
    for i in tqdm(validation_df['userId'].unique()):
        temp = validation_df[validation_df["userId"] == i]["movieId"].tolist()
        purchases_val.append(temp)
    # print(purchases_val)
    # print(len(purchases_val))
    return purchases_train,purchases_val,products_dict

# 每个商品的word2vec embeddings
def word2vecEmbedding(purchases_train):
    # 训练word2vec模型
    model = Word2Vec(window=5,
                     sg=1,
                     hs=0,
                     negative=10,  # for negative sampling
                     alpha=0.03,
                     min_alpha=0.0007,
                     seed=14,
                     workers=8)
    model.build_vocab(purchases_train, progress_per=200)
    model.train(purchases_train, total_examples=model.corpus_count,
                epochs=10, report_delay=1)
    model.init_sims(replace=True) # 模型的内存效率更高
    model.save('../data/word2vec_model')
    return model # 查看model相关参数

"""
gensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, size=100, 
alpha=0.025, window=5, min_count=5, 
max_vocab_size=None, sample=0.001, seed=1, 
workers=3, min_alpha=0.0001, sg=0, 
hs=0, negative=5, ns_exponent=0.75, 
cbow_mean=1, hashfxn=<built-in function hash>, 
iter=5, null_word=0, trim_rule=None, 
sorted_vocab=1, batch_words=10000, 
compute_loss=False, callbacks=(), max_final_vocab=None)

参数解释：
·  sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
·  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
·  size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
·  window：表示当前词与预测词在一个句子中的最大距离是多少
·  alpha: 是学习速率
·  seed：用于随机数发生器。与初始化词向量有关。
·  min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
·  max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
·  sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
·  workers:参数控制训练的并行数,默认是3个核心。
·  hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
·  negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
·  cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
·  hashfxn： hash函数来初始化权重。默认使用python的hash函数
·  iter： 迭代次数，默认为5
·  trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
·  sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
·  batch_words：每一批的传递给线程的单词的数量，默认为10000

# 存储和载入模型
model.save('/tmp/mymodel')
new_model = gensim.models.Word2Vec.load('/tmp/mymodel')

# 该方法将模型内部的 NumPy 矩阵从硬盘载入到虚拟内存。
# 另外，可以使用如下的方法载入原生 C 工具生成的模型，文本和二进制形式的均可。
model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
## using gzipped/bz2 input works too, no need to unzip:
model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)

# 在线训练和恢复训练
## 高级用户可以载入模型后用更多的预料对其进行训练，你可能要对参数 total_words 进行调整，取决于希望达到的学习率。
model = gensim.models.Word2Vec.load('/tmp/mymodel')
model.train(more_sentences)
"""

def visualize_word2vec(model):
    X = model[model.wv.vocab]
    print(X.shape)
    cluster_embedding = umap.UMAP(n_neighbors=30,
                                  min_dist=0.0,
                                  n_components=2,
                                  random_state=42).fit_transform(X)
    plt.figure(figsize=(10, 9))
    plt.scatter(cluster_embedding[:, 0],
                cluster_embedding[:, 1],
                s=3,
                cmap='Spectral')
    plt.savefig('../data/model.png')

# 将一个商品的向量(n)作为输入，返回前6个相似的商品
def similar_products(v, model, products_dict, n = 6):
    # 为输入向量提取最相似的商品
    ms = model.similar_by_vector(v, topn= n+1)[1:]
    # 提取相似产品的名称和相似度评分
    new_ms = []
    for j in ms:
        pair = (products_dict[j[0]][0], j[1])
        new_ms.append(pair)
    return new_ms

# 我们想根据他或她过去的多次购买来推荐商品呢?  这个比较有意思。。。
# 取用户迄今为止购买的所有商品的向量的平均值，并使用这个结果向量找到类似的商品
# 使用下面的函数，它接收一个商品id列表，并返回一个100维的向量，它是输入列表中商品的向量的平均值
def aggregate_vectors(products,model):
    product_vec = []
    for i in products:
        try:
            product_vec.append(model[i])
        except KeyError:
            continue
    return np.mean(product_vec, axis=0)

def run(input_file_user,input_file_item):
    purchases_train,purchases_val,products_dict = data_process(input_file_user,input_file_item)

    # t1 = time.time()
    # model = word2vecEmbedding(purchases_train)
    # t2 = time.time()
    # print(t2-t1)

    model = Word2Vec.load('../data/word2vec_model')

    # 可视化
    visualize_word2vec(model)

    # 通过传递商品编号为'1'的商品
    print(similar_products(v = model['1'],model = model, products_dict = products_dict))

    # 为了验证目的，我们已经创建了一个单独的购买序列列表。现在刚好可以利用它
    # 用户购买的第一个商品列表的长度为314。我们将把这个验证集的商品序列传递给aggregate_vectors函数
    print(len(purchases_val[0]))

    # 函数返回了一个100维的数组。这意味着函数运行正常。现在我们可以用这个结果得到最相似的商品:
    print(aggregate_vectors(purchases_val[0],model).shape)

    # 结果，我们的系统根据用户的整个购买历史推荐了6款商品。此外，你也可以根据最近几次购买情况来进行商品推荐。
    # 下面我只提供了最近购买的10种商品作为输入:
    res = similar_products(v = aggregate_vectors(purchases_val[0],model), model = model, products_dict = products_dict)
    print(res)

if __name__ == "__main__":
    run('../data/ratings.txt','../data/movies.txt')