这是一个简单的推荐系统，使用 TensorFlow 和 Python 3 开发。

# 1 数据集描述

## 一、用户数据

数据格式：UserID::Gender::Age::Occupation::Zip-code

Gender is denoted by a "M" for male and "F" for female

Age is chosen from the following ranges:

    1: "Under 18"
    18: "18-24"
    25: "25-34"
    35: "35-44"
    45: "45-49"
    50: "50-55"
    56: "56+"

Occupation is chosen from the following choices:

    0: "other" or not specified
    1: "academic/educator"
    2: "artist"
    3: "clerical/admin"
    4: "college/grad student"
    5: "customer service"
    6: "doctor/health care"
    7: "executive/managerial"
    8: "farmer"
    9: "homemaker"
    10: "K-12 student"
    11: "lawyer"
    12: "programmer"
    13: "retired"
    14: "sales/marketing"
    15: "scientist"
    16: "self-employed"
    17: "technician/engineer"
    18: "tradesman/craftsman"
    19: "unemployed"
    20: "writer"



## 二、电影数据

数据格式：MovieID::Title::Genres

Titles are identical to titles provided by the IMDB (including year of release)

Genres are pipe-separated and are selected from the following genres:
    Action
    Adventure
    Animation
    Children's
    Comedy
    Crime
    Documentary
    Drama
    Fantasy
    Film-Noir
    Horror
    Musical
    Mystery
    Romance
    Sci-Fi
    Thriller
    War
    Western

# 2 问题改进
1. 加上年份信息，看看是否影响推荐效果。

2. GBDT（lightGBM，xgboost）。

3. 电影类型不使用 sum，使用其他方式来描述电影特征，获得更好地效果。

4. 尝试随机采样。

1. 将用户的所有属性都设置为 32 特征，而不是有的是 32， 有的是 16。

2. 将电影特征的dropout加到拼接好的电影特征上，而不是加到 title 的特征上。

3. 拼接的用户矩阵没有使用 dropout，试着将用户特征和电影特征都加上 dropout 看看最终的性能是否提升。

4. 尝试使用 tensorflow 自己的 batch 方法。

5. 作者的训练方式有点问题，他获取 batch 的时候是按照顺序来获取样本的，尝试随机获取样本，看看最终的性能变化。

## 问题
1. user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200]) 为什么还要把最后一层变形？
答：因为最后要计算得分，得分等于用户特征矩阵和电影特征矩阵的乘积。所以要将用户矩阵和电影矩阵拉平。

2. Session 在 tensorflow 中的作用？
答：Session 为创建的 tensor 分配计算资源。

3. Graph 在 tensorflow 中的作用？
答：Graph 中包含运算和涉及到的 tensor。它可以明确得表现出各个 tensor 之间的运算关系。

# 3 推荐功能
使用卷积神经网络，并利用MovieLens数据集完成电影推荐的任务。
实现的推荐功能如下：
 - 1、指定用户和电影进行评分
 - 2、推荐同类型的电影
 - 3、推荐您喜欢的电影
 - 4、看过这个电影的人还看了（喜欢）哪些电影

## 指定用户和电影进行评分
给用户234，电影1401的评分是：4.27963877

## 推荐同类型的电影
您看的电影是：[1401 'Ghosts of Mississippi (1996)' 'Drama']

以下是给您的推荐：

3385

[3454 'Whatever It Takes (2000)' 'Comedy|Romance']

707

[716 'Switchblade Sisters (1975)' 'Crime']

2351

[2420 'Karate Kid, The (1984)' 'Drama']

2189

[2258 'Master Ninja I (1984)' 'Action']

2191

[2260 'Wisdom (1986)' 'Action|Crime']

## 推荐您喜欢的电影
以下是给您的推荐（用户234）：

1642

[1688 'Anastasia (1997)' "Animation|Children's|Musical"]

994

[1007 'Apple Dumpling Gang, The (1975)' "Children's|Comedy|Western"]

667

[673 'Space Jam (1996)' "Adventure|Animation|Children's|Comedy|Fantasy"]

1812

[1881 'Quest for Camelot (1998)' "Adventure|Animation|Children's|Fantasy"]

1898

[1967 'Labyrinth (1986)' "Adventure|Children's|Fantasy"]

## 看过这个电影的人还看了（喜欢）哪些电影
您看的电影是：[1401 'Ghosts of Mississippi (1996)' 'Drama']

喜欢看这个电影的人是：[[5782 'F' 35 0]

 [5767 'M' 25 2]
 
 [3936 'F' 35 12]
 
 [3595 'M' 25 0]
 
 [1696 'M' 35 7]
 
 [2728 'M' 35 12]
 
 [763 'M' 18 10]
 
 [4404 'M' 25 1]
 
 [3901 'M' 18 14]
 
 [371 'M' 18 4]
 
 [1855 'M' 18 4]
 
 [2338 'M' 45 17]
 
 [450 'M' 45 1]
 
 [1130 'M' 18 7]
 
 [3035 'F' 25 7]
 
 [100 'M' 35 17]
 
 [567 'M' 35 20]
 
 [5861 'F' 50 1]
 
 [4800 'M' 18 4]
 
 [3281 'M' 25 17]]
 
喜欢看这个电影的人还喜欢看：

1779

[1848 'Borrowers, The (1997)' "Adventure|Children's|Comedy|Fantasy"]

1244

[1264 'Diva (1981)' 'Action|Drama|Mystery|Romance|Thriller']

1812

[1881 'Quest for Camelot (1998)' "Adventure|Animation|Children's|Fantasy"]

1742

[1805 'Wild Things (1998)' 'Crime|Drama|Mystery|Thriller']

2535

[2604 'Let it Come Down: The Life of Paul Bowles (1998)' 'Documentary']
