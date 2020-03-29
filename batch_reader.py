# ==============================================================================
# -*- coding: utf-8 -*-
# @Time    : 2020-03-08 AM 10:30
# @Author  : deymm
# @Email   : deymm671@126.com
# ==============================================================================

"""数据读取"""

from collections import namedtuple
import Queue
from random import shuffle
from threading import Thread
import time

import numpy as np
import tensorflow as tf
import parameter_config
import data
import os

ModelInput = namedtuple('ModelInput',
                        'index_id target enc_inputs enc_input_len')

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100


class Batcher(object):
    """Batch reader with shuffling and bucketing support."""

    def __init__(self, data_path, vocab, index_key, target_key, 
        sentence_key, hps, bucketing=False, truncate_input=False):
        """Batcher constructor, 数据读取器

        Args:
          data_path: data source file pattern.
          vocab: class vocab
          index_key: string, the name of index_key
          target_key: string, the name of target_key
          sentence_key: string, the name of sentence_key
          hps: model hyperparameters.
          bucketing: Whether bucket sentence of similar length into the same batch.
          truncate_input: Whether to truncate input that is too long. Alternative is
            to discard such examples.
        """
        self._data_path = data_path
        self._vocab = vocab
        self._index_key = index_key
        self._target_key = target_key
        self._sentence_key = sentence_key
        self._hps = hps
        self._bucketing = bucketing
        self._truncate_input = truncate_input
        self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
        self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)

        self._input_threads = []
        for _ in xrange(1):
            self._input_threads.append(Thread(target=self._FillInputQueue))
            self._input_threads[-1].daemon = True
            self._input_threads[-1].start()

        if hps.mode == 'train':
            self._bucketing_threads = []
            for _ in xrange(1):
                self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
                self._bucketing_threads[-1].daemon = True
                self._bucketing_threads[-1].start()
            self._watch_thread = Thread(target=self._WatchThreads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def NextTrainBatch(self):
        """Returns a batch of inputs for model to train or evaluation.

        Returns:
          index_batch: A batch of index_key [batch_size]
          enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
          target_batch: A batch of targets [batch_size, 1].
          enc_input_len: encoder input lengths of the batch.
        """
        enc_batch = np.zeros(
            (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
        enc_input_lens = np.zeros(
            (self._hps.batch_size), dtype=np.int32)
        target_batch = np.zeros( (self._hps.batch_size, 1), dtype=np.float32)
        index_batch = ['None'] * self._hps.batch_size

        buckets = self._bucket_input_queue.get()
        for i in xrange(self._hps.batch_size):
            (index_id, target, enc_inputs, enc_input_len) = buckets[i]

            index_batch[i] = index_id
            target_batch[i][0] = target
            enc_batch[i][:] = enc_inputs
            enc_input_lens[i] = enc_input_len

        return (index_batch, target_batch, enc_batch, enc_input_lens)

    def NextEvalBatch(self):
        """Returns a batch of inputs for model to evaluation or decode.

        Returns:
          index_batch: A batch of index_key [batch_size]
          enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
          target_batch: A batch of targets [batch_size, 1].
          enc_input_len: encoder input lengths of the batch.
          batch_lens: the length of index_batch
        """
        enc_batch = np.zeros(
            (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
        enc_input_lens = np.zeros(
            (self._hps.batch_size), dtype=np.int32)
        target_batch = np.zeros( (self._hps.batch_size, 1), dtype=np.float32)
        index_batch = ['None'] * self._hps.batch_size

        batch_lens = 0
        for i in xrange(self._hps.batch_size):
            if self._input_queue.empty():
                break
            inputs = self._input_queue.get()
            (index_id, target, enc_inputs, enc_input_len) = inputs
            batch_lens += 1

            index_batch[i] = index_id
            target_batch[i][0] = target
            enc_batch[i][:] = enc_inputs
            enc_input_lens[i] = enc_input_len

        return (index_batch, target_batch, enc_batch, enc_input_lens, batch_lens)

    def _FillInputQueue(self):
        """逐行填充输入队列"""
        pad_id = self._vocab.WordToId(parameter_config.PAD_TOKEN)
        if self._hps.mode == 'train':
            input_gen = self._TextGenerator(data.ExampleGen(os.path.join(self._data_path,'*')))
        else:
            input_gen = self._TextGenerator(data.ExampleGen(os.path.join(self._data_path,'*'),1))

        while True:
            try:
                (index_id, target, sentence) = input_gen.next()
            except (GeneratorExit, StopIteration):
                break

            enc_inputs = data.GetWordIds(sentence.strip(), self._vocab)
            target = int(target)

            # Filter out too-short input
            if (len(enc_inputs) < self._hps.min_input_len):
                # tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d',
                #                   len(enc_inputs), len(dec_inputs))
                continue

            # If we're not truncating input, throw out too-long input
            if not self._truncate_input:
                if (len(enc_inputs) > self._hps.enc_timesteps):
                    # tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d',
                    #                  len(enc_inputs), len(dec_inputs))
                    continue
            # If we are truncating input, do so if necessary
            else:
                if len(enc_inputs) > self._hps.enc_timesteps:
                    enc_inputs = enc_inputs[:self._hps.enc_timesteps]

            enc_input_len = len(enc_inputs)

            # Pad if necessary
            while len(enc_inputs) < self._hps.enc_timesteps:
                enc_inputs.append(pad_id)

            element = ModelInput(index_id, target, enc_inputs, enc_input_len)
            self._input_queue.put(element)

    def _FillBucketInputQueue(self):
        """按batch_size长度填充Bucket输入队列"""
        while True:
            inputs = []
            for _ in xrange(self._hps.batch_size * BUCKET_CACHE_BATCH):
                inputs.append(self._input_queue.get())
            if self._bucketing:
                inputs = sorted(inputs, key=lambda inp: inp.enc_input_len)

            batches = []
            for i in xrange(0, len(inputs), self._hps.batch_size):
                batches.append(inputs[i:i + self._hps.batch_size])
            shuffle(batches)
            for b in batches:
                self._bucket_input_queue.put(b)

    def _WatchThreads(self):
        """监听器，监听输入线程"""
        while True:
            time.sleep(30)
            input_threads = []
            for t in self._input_threads:
                if t.is_alive():
                    input_threads.append(t)
                else:
                    # tf.logging.error('Found input thread dead.')
                    new_t = Thread(target=self._FillInputQueue)
                    input_threads.append(new_t)
                    input_threads[-1].daemon = True
                    input_threads[-1].start()
            self._input_threads = input_threads

            bucketing_threads = []
            for t in self._bucketing_threads:
                if t.is_alive():
                    bucketing_threads.append(t)
                else:
                    # tf.logging.error('Found bucketing thread dead.')
                    new_t = Thread(target=self._FillBucketInputQueue)
                    bucketing_threads.append(new_t)
                    bucketing_threads[-1].daemon = True
                    bucketing_threads[-1].start()
            self._bucketing_threads = bucketing_threads

    def _TextGenerator(self, example_gen):
        """Generates index target and sentence from tf.Example."""
        while True:
            e = example_gen.next()
            try:
                index_text = self._GetExFeatureText(e, self._index_key)
                target_text = self._GetExFeatureText(e, self._target_key)
                sentence_text = self._GetExFeatureText(e, self._sentence_key)
            except ValueError:
                # tf.logging.error('Failed to get sentence from example')
                continue

            yield (index_text, target_text, sentence_text)

    def _GetExFeatureText(self, ex, key):
        """根据关键词从tf.Example获得相应的数据
           关键词有 index target sentence

        Args:
          ex: tf.Example.
          key: string, key of the feature to be extracted.
        Returns:
          feature: a feature text extracted.
        """
        return ex.features.feature[key].bytes_list.value[0]
