# ==============================================================================
# -*- coding: utf-8 -*-
# @Time    : 2020-03-07 PM 7:21
# @Author  : deymm
# @Email   : deymm671@126.com
# ==============================================================================

# unknow词
UNKNOWN_TOKEN = '<unknow>'
# Pad词
PAD_TOKEN = '<pad>'
# 分隔符
SEP_TOKEN = '##'

# 原始数据路径
ORIGIN_DIR = 'data/original/'
# train set path
TRAIN_DIR = 'data/train/'
# dev set path
DEV_DIR = 'data/dev/'
# test set path
TEST_DIR = 'data/test/'
# 生成的词表路径
VOCAB_DIR = 'data/vocabulary/'
# 生成词表的文件名
VOCAB_FILE_NAME = 'vocab.txt'
# 解码目录
DECODE_DIR = 'data/dev/'
# 解码文件存储目录 
DECODE_STORE_DIR = 'data/decode/'
# 解码存储文件名
DECODE_STORE_FILE = 'decode.txt'
# 各数据集样本比例，相加为1
SAMPLE_RATE = [0.7, 0.2, 0.1]
# 评估集列表
EVALUATION_LIST = [TRAIN_DIR, DEV_DIR, TEST_DIR]
# 评估集, 0:train set; 1:develop set; 2:test set
EVALUATION_SET = EVALUATION_LIST[1]
# 模型保存路径
CKPT_PATH = 'ckpt'
# 模型名称
MODEL_NAME = 'model'

# 词表大小
VOCAB_SIZE = 10000
# eval_step模式下的sleep time
SLEEP_TIME = 2
# train模式下loss输出频率
LOSS_PRINT_STEP = 10
# 模型文件存储频率
MODEL_SAVE_STEP = 100
# 训练step数
TRAIN_STEP = 2000

# 模型超参数
# batch大小
BATCH_SIZE = 64
# encoder序列长度
ENC_TIMESTEPS = 50
# word embedding维度
EMB_DIM = 128
# 输入句子的最小长度
MIN_INPUT_LEN = 2
# 隐藏层维度
NUM_HIDDEN = 64
# encoder层数
ENC_LAYERS = 2
# 最小学习率
MIN_LR = 0.00001
# 学习率
LR = 0.0001
# 最大梯度
MAX_GRAD_NORM = 10
# 网络层dropout保留率
NN_KEEP_PROB = 0.5
# Embedding层dropout保留率
EMB_KEEP_PROB = 0.5
# 学习率衰减
LR_DECAY = 0.99
# 学习率衰减频率
LR_DECAY_STEP = 1000
# 移动平均衰减率
EMA_RATE = 0.99
