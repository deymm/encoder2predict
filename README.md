# 基于循环神经网络（RNN）的序列数据概率（或分类）预测
在二分类任务中，使用RNN的Encoder层（LSTM）对序列数据进行概率（或分类）预测

出发点：在信贷业务中，存在用户的序列化数据，希望将该序列化数据进行Embedding从而预测逾期，因此搭了个基于二分类任务的框架（也可用于其它场景下的二分类任务）

实验数据集：因为数据的敏感性，所以这里用了评论情感分类数据作为测试数据([https://www.cs.cornell.edu/people/pabo/movie-review-data/](https://www.cs.cornell.edu/people/pabo/movie-review-data/) 上的[sentence polarity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)，包含正负面评论各5331条)

-------------------

## 使用说明
#### Step 1 数据预处理

```cmd
python preprocess.py 
```

数据预处理过程包括：

> 1.解码文件成txt文件

> 2.生成原始文件（正负数据集放在一起）

> 3.生成词汇表文件

> 4.生成数据集划分文件（Train、Dev、Test）

#### Step 2 模型框架

模型框架部分在文件seq2seq_model.py中

使用了RNN的Encoder层作为主体框架，包括：

> 1.对输入序列进行Embedding

> 2.使用LSTM作为基本单元，构建多层Encoder层（这里用了两层前向神经网络）

> 3.对输入的Embedding序列数据和LSTM进行随机失活(dropout)，防止过拟合

> 4.对Encoder最后层最后单元的隐藏层输出加一层NN层，通过sigmoid预测概率（类别）（相当于对最后隐层输出做了逻辑回归）

#### Step 3 数据读取

使用batch_reader.py进行数据读取：

> 1.设置了input队列、bucket队列和监视器

> 2.当未达到训练停止条件时，监视器监控两个队列不断读取数据（input读取的数据传递给bucket队列，构成输出的batch数据）

#### Step 4 模型训练

```cmd
python seq2seq_mian.py train
```

训练过程中使用移动平均和学习率指数衰减

#### Step 5 模型评估
**评估指标：**

模型评估函数在文件evaluation_function.py中

> 1.Accuracy

> 2.AUC

> 3.KS

**提供了两种评估模式：**

1.等模型训练完成后对Dev集合进行评估

```cmd
python seq2seq_mian.py eval
```

2.在模型训练的过程中，一边训练一边评估，输出每个step下的评估值

```cmd
python seq2seq_mian.py eval_step
```

#### Step 6 模型预测输出

```cmd
python seq2seq_mian.py decode
```

输出数据存储格式：index_label,target,logits（NN层输出）,predict（对NN层输出做sigmoid变化后的概率值）

#### Step 7 模型设置

模型设置参数在parameter_config.py文件中
