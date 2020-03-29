# ==============================================================================
# -*- coding: utf-8 -*-
# @Time    : 2020-03-07 PM 7:21
# @Author  : deymm
# @Email   : deymm671@126.com
# ==============================================================================

"""以情感分析语料作为基础，构建实验语料和词表"""
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')
import parameter_config
import collections
import numpy as np
from random import shuffle

# 原始输入文件地址
ORIGIN_NEG_PATH = 'data/rt-polarity.neg'
ORIGIN_POS_PATH = 'data/rt-polarity.pos'

# 解码后的文件地址
TEXT_NEG_PATH = 'data/neg.txt'
TEXT_POS_PATH = 'data/pos.txt'

# 需要处理的原始文档路径
DATA_DIR_NAME = 'data'
ROOT_PATH = os.getcwd()
ORIGINAL_FILE_NAME = 'original.txt'
TRAIN_FILE_NAME = 'train.txt'
DEV_FILE_NAME   = 'dev.txt'
TEST_FILE_NAME  = 'test.txt'



def DecodeFile(infile, outfile):
    """将文件的编码从'Windows-1252'转为Unicode

    Args:
      infile: string, 输入文件路径
      outfile: string, 输出文件路径

    Returns:

    """
    with open(infile, 'r') as f:
        txt = f.read().decode('Windows-1252')
    with open(outfile, 'w') as f:
        f.write(txt)


def GenerateFormatData(neg_path, pos_path):
    """生成需要的语料格式 index+target_label+sentence

    Args:
      neg_path: string, 负样本文件地址
      pos_path: string, 正样本文件地址

    Returns:

    """
    # 判断存储生成目标格式文件的目录是否存在，不存在则先生成
    store_dir_path = os.path.join(ROOT_PATH,parameter_config.ORIGIN_DIR)
    if not os.path.exists(store_dir_path):
        os.mkdir(store_dir_path)

    # 将数据组装成 index+target_label+sentence形式
    index_label = 0
    sentence_list = []

    with open(neg_path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            sentence = str(index_label) + parameter_config.SEP_TOKEN + str(1) + parameter_config.SEP_TOKEN + line.strip()
            sentence_list.append(sentence)
            index_label += 1
    with open(pos_path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            sentence = str(index_label) + parameter_config.SEP_TOKEN + str(0) + parameter_config.SEP_TOKEN + line.strip()
            sentence_list.append(sentence)
            index_label += 1
    
    # 将数据打乱并写入文件中
    shuffle(sentence_list)

    with open(os.path.join(parameter_config.ORIGIN_DIR,ORIGINAL_FILE_NAME), 'w') as f:
        for sentence in sentence_list:
            f.write(sentence + '\n')

    
def CutSet(original_path, cut_rate):
    """将数据集按比例切分成train,dev,test

    Args:
      original_path: string, 原始语料文件目录
      cut_rate: list, 3维list，表示train,dev,test样本占比

    Returns:

    """
    # 判断存储的目录是否存在，不存在则先生成
    train_dir_path = os.path.join(ROOT_PATH,parameter_config.TRAIN_DIR)
    dev_dir_path   = os.path.join(ROOT_PATH,parameter_config.DEV_DIR)
    test_dir_path  = os.path.join(ROOT_PATH,parameter_config.TEST_DIR)
    if not os.path.exists(train_dir_path):
        os.mkdir(train_dir_path)
    if not os.path.exists(dev_dir_path):
        os.mkdir(dev_dir_path)
    if not os.path.exists(test_dir_path):
        os.mkdir(test_dir_path)

    # 划分数据存储
    data = [[], [], []]
    
    # 累计比例 cut_rate = [0.7,0.2,0.1] 则cumsum_rate=[0.7,0.9,1.0]
    cumsum_rate = np.cumsum(cut_rate)

    # 使用轮盘赌法划分数据集
    with open(original_path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            index = int(np.searchsorted(cumsum_rate, np.random.rand(1) * 1.0))
            data[index].append(line.strip())
    
    # Train数据存储
    with open(os.path.join(parameter_config.TRAIN_DIR, TRAIN_FILE_NAME), 'w') as f:
        for sentence in data[0]:
            f.write(sentence + '\n')
    # Develop数据存储
    with open(os.path.join(parameter_config.DEV_DIR, DEV_FILE_NAME), 'w') as f:
        for sentence in data[1]:
            f.write(sentence + '\n')
    # Test数据存储
    with open(os.path.join(parameter_config.TEST_DIR, TEST_FILE_NAME), 'w') as f:
        for sentence in data[2]:
            f.write(sentence + '\n')


def CreateVocab(original_path, vocab_path):
    """构建词汇表

    Args:
      original_path: string, 负样本文件地址
      vocab_path: string, 词汇表存储地址

    Returns:
    """
    
    # 判断存储生成词表的目录是否存在，不存在则先生成
    vocab_dir_path = os.path.join(ROOT_PATH,parameter_config.VOCAB_DIR)
    if not os.path.exists(vocab_dir_path):
        os.mkdir(vocab_dir_path)
    # 存放遍历的所有单词
    word_list = []
    # 数据读取
    good_line_cnt = 0
    bad_line_cnt = 0 
    with open(original_path, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            sentence = line.strip().split(parameter_config.SEP_TOKEN)
            if len(sentence) != 3:
                bad_line_cnt += 1
                continue
            
            good_line_cnt += 1
            words = (sentence[-1]).strip().split()
            word_list.extend(words)

    sys.stderr.write('Good line: %d\n Bad line: %d\n' % (good_line_cnt,bad_line_cnt))

    # 统计单词出现的次数并排序
    words_counter = collections.Counter(word_list)
    words_sorted = sorted(words_counter.items(), key=lambda x: x[1], reverse=True)
    # 添加 <unknow> 和 <pad>
    word_list = [word[0] for word in words_sorted]
    word_list = [parameter_config.PAD_TOKEN, parameter_config.UNKNOWN_TOKEN] + word_list
    # 将词表写入文件
    with open(vocab_path, 'w') as f:
        for word in word_list:
            f.write(word + '\n')



if __name__ == '__main__':
    # 解码文件成txt
    DecodeFile(ORIGIN_NEG_PATH, TEXT_NEG_PATH)
    DecodeFile(ORIGIN_POS_PATH, TEXT_POS_PATH)
    
    # 各文件目录
    original_path = os.path.join(parameter_config.ORIGIN_DIR,ORIGINAL_FILE_NAME)
    vocab_path    = os.path.join(parameter_config.VOCAB_DIR, parameter_config.VOCAB_FILE_NAME)

    # 生成原始文件
    GenerateFormatData(TEXT_NEG_PATH, TEXT_POS_PATH)
    # 生成词汇表文件
    CreateVocab(original_path, vocab_path)
    # 生成数据集划分文件
    CutSet(original_path, parameter_config.SAMPLE_RATE)

    