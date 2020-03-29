# 基于循环神经网络（RNN）的序列数据概率（或分类）预测
在二分类任务中，使用RNN的Encoder层（LSTM）对序列数据进行概率（或分类）预测

出发点：在信贷业务中，存在用户的序列化数据，希望将该序列化数据进行Embedding从而预测逾期，因此搭了个基于二分类任务的框架（也可用于其它场景下的二分类任务）

实验数据集：因为数据的敏感性，所以这里用了评论情感分类数据作为测试数据(https://www.cs.cornell.edu/people/pabo/movie-review-data/上的sentence polarity dataset v1.0，包含正负面评论各5331条)

具体说明详情可移步我的知乎专栏文章

# 使用说明
## Step 1
