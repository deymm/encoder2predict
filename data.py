# ==============================================================================
# -*- coding: utf-8 -*-
# @Time    : 2020-03-07 PM 8:21
# @Author  : deymm
# @Email   : deymm671@126.com
# ==============================================================================
import parameter_config
import glob
import random
import struct
import sys

from tensorflow.core.example import example_pb2

class Vocab(object):
    """读取词表生成词表字典，用于匹配词和编码之间的关系"""

    def __init__(self, vocab_file, max_size):
        """初始化

        Args:
          vocab_file: string, 词表目录
          max_size: int, 选定词表长度

        Return:
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 1:
                    sys.stderr.write('Vocabulary Bad line: %s\n' % line)
                    continue
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                self._word_to_id[pieces[0]] = self._count
                self._id_to_word[self._count] = pieces[0]
                self._count += 1
                if self._count >= max_size:
                    break

                
    def WordToId(self, word):
        """根据词表把word转换成id """
        if word not in self._word_to_id:
            return self._word_to_id[parameter_config.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self, word_id):
        """根据词表把id转换成word"""
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        """返回词表类的长度"""
        return self._count



def Pad(ids, pad_id, length):
    """对输入sentence进行填充或者截断

    Args:
      ids: list, 转成id的sentence list
      pad_id: int, 填充词的id
      length: int, 限定输入句子长度

    Returns:
      list, 填充或者截断后的ids list
    """
    assert pad_id is not None
    assert length is not None

    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return ids + a
    else:
        return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
    """把word序列 转换成id list，word之间用空格分隔

    Args:
      text: string, input sentence
      vocab: Vocab object
      pad_len: int, 期望句子长度
      pad_id: int, 填充词的id

    Returns:
      list, 用词的id组成的句子
    """
    ids = []
    for w in text.split():
        i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(parameter_config.UNKNOWN_TOKEN))
    if pad_len is not None:
        return Pad(ids, pad_id, pad_len)
    return ids


def ExampleGen(data_path, num_epochs=None):
    """把从目录文件下的每行数据转换成tf.Examples格式，并生成generator（生成器）
       关键词为：index_key, 目标 和 输入序列

    Args:
      data_path: 数据目录名
      num_epochs: 每个文件数据读取次数. None则视为无限读取.

    Yields:
      tf.Example格式的生成器
    """
    epoch = 0
    while True:
        if num_epochs is not None and epoch >= num_epochs:
            break
        filelist = glob.glob(data_path)
        assert filelist, 'Empty filelist.'
        random.shuffle(filelist)
        for file in filelist:
            with open(file, 'r') as f:
                f_lines = f.readlines()
                while True:
                    for line in f_lines:
                        if line is None: break
                        tf_example = example_pb2.Example()

                        contents = line.strip().split(parameter_config.SEP_TOKEN)
                        ## python 3
                        # tf_example.features.feature["index"].bytes_list.value.extend([bytes(contents[0], encoding='utf-8')])
                        # tf_example.features.feature["target"].bytes_list.value.extend([bytes(contents[1], encoding='utf-8')])
                        # tf_example.features.feature["sentence"].bytes_list.value.extend([bytes(contents[2], encoding='utf-8')])
                        ## python 2
                        tf_example.features.feature["index"].bytes_list.value.extend([bytes(contents[0])])
                        tf_example.features.feature["target"].bytes_list.value.extend([bytes(contents[1])])
                        tf_example.features.feature["sentence"].bytes_list.value.extend([bytes(contents[2])])

                        yield tf_example
                    break

        epoch += 1
