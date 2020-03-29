# ==============================================================================
# -*- coding: utf-8 -*-
# @Time    : 2020-03-08 PM 4:50
# @Author  : deymm
# @Email   : deymm671@126.com
# ==============================================================================
"""模型执行文件"""
import time
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import tensorflow as tf
import parameter_config
import data
import batch_reader
import seq2seq_model
import evaluation_function


def _Train(model, data_batcher, max_run_steps):
    """模型训练"""
    with tf.Session() as sess:
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for step in range(max_run_steps):
            (index_batch, target_batch, enc_batch, enc_input_lens) = data_batcher.NextTrainBatch()
            loss, global_step, _ = model.run_train_step(sess, target_batch, enc_batch, enc_input_lens, 
                parameter_config.NN_KEEP_PROB, parameter_config.EMB_KEEP_PROB)
            # every step 输出loss
            if (step+1) % parameter_config.LOSS_PRINT_STEP == 0:
                print ('step {},loss is {}'.format(step+1, loss))
            # 保存模型
            if (step+1) % parameter_config.MODEL_SAVE_STEP == 0:
                saver.save(sess, os.path.join(parameter_config.CKPT_PATH, parameter_config.MODEL_NAME), global_step)


def _Eval(model, data_batcher):
    """训练完数据后，一次性评估
       模型评估(截取法，最后小于batch_size的数据直接舍弃)
       读取最近一次的模型参数进行评估
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        model.build_graph()
        saver = tf.train.Saver()
        # 加载最新的模型
        ckpt = tf.train.get_checkpoint_state(parameter_config.CKPT_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)

        predict_list = []
        target_list = []

        while True:
            (index_batch, target_batch, enc_batch, enc_input_lens, batch_lens) = data_batcher.NextEvalBatch()
            if batch_lens != parameter_config.BATCH_SIZE:
                break
            loss, predict, global_step = model.run_eval_step(sess, target_batch, enc_batch, enc_input_lens, 1.0, 1.0)
            predict_list.extend(list(np.reshape(predict,parameter_config.BATCH_SIZE)))
            target_list.extend(list(np.reshape(target_batch,parameter_config.BATCH_SIZE)))

        acc = evaluation_function.calculate_acc(target_list,predict_list)
        auc = evaluation_function.calculate_auc(target_list,predict_list)
        ks = evaluation_function.calculate_ks(target_list,predict_list)
        print ('step:{} acc:{} auc:{} ks :{}'.format(global_step,acc,auc,ks))


def _Eval_Step(model):
    """边训练边评估
       模型评估(截取法，最后小于batch_size的数据直接舍弃)
       读取最近一次的模型参数进行评估
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        model.build_graph()
        saver = tf.train.Saver()
        # 加载最新的模型
        while True:
            data_batcher = batch_reader.Batcher(parameter_config.EVALUATION_SET, model._vocab, 'index', 
                'target', 'sentence', model._hps, bucketing=False, truncate_input=True)
            ckpt = tf.train.get_checkpoint_state(parameter_config.CKPT_PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)

            predict_list = []
            target_list = []

            while True:
                (index_batch, target_batch, enc_batch, enc_input_lens, batch_lens) = data_batcher.NextEvalBatch()
                if batch_lens != parameter_config.BATCH_SIZE:
                    break
                loss, predict, global_step = model.run_eval_step(sess, target_batch, enc_batch, enc_input_lens, 1.0, 1.0)
                predict_list.extend(list(np.reshape(predict,parameter_config.BATCH_SIZE)))
                target_list.extend(list(np.reshape(target_batch,parameter_config.BATCH_SIZE)))

            acc = evaluation_function.calculate_acc(target_list,predict_list)
            auc = evaluation_function.calculate_auc(target_list,predict_list)
            ks = evaluation_function.calculate_ks(target_list,predict_list)
            print ('step:{} acc:{} auc:{} ks :{}'.format(global_step,acc,auc,ks))
            time.sleep(parameter_config.SLEEP_TIME)


def _Decode(model, data_batcher, store_path):
    """模型预测结果存储"""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        model.build_graph()
        saver = tf.train.Saver()
        # 加载最新的模型
        ckpt = tf.train.get_checkpoint_state(parameter_config.CKPT_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        sentence_list = []
        while True:
            (index_batch, target_batch, enc_batch, enc_input_lens, batch_lens) = data_batcher.NextEvalBatch()
            if batch_lens != 1:
                break
            logits, predict, global_step = model.run_decode_step(sess, target_batch, enc_batch, enc_input_lens, 1.0, 1.0)
            # 输出结果为 index_label,target,logits,predict
            sentence = str(index_batch[0])+','+str(int(target_batch[0][0]))+','+str(logits[0][0])+','+str(predict[0][0])
            sentence_list.append(sentence)

        with open(store_path, 'w') as f:
            for sentence in sentence_list:
                f.write(sentence + '\n')

         
def main(mode_type):
    # 读取词表
    vocab = data.Vocab(os.path.join(parameter_config.VOCAB_DIR, parameter_config.VOCAB_FILE_NAME), parameter_config.VOCAB_SIZE)
    batch_size = parameter_config.BATCH_SIZE
    if mode_type == 'decode':
        batch_size = 1

    # 设置模型超参数
    hps = seq2seq_model.HParams(
        mode=mode_type,  # train, eval, decode
        batch_size = batch_size,
        enc_timesteps = parameter_config.ENC_TIMESTEPS,
        emb_dim = parameter_config.EMB_DIM,
        min_input_len = parameter_config.MIN_INPUT_LEN,
        num_hidden = parameter_config.NUM_HIDDEN,
        enc_layers = parameter_config.ENC_LAYERS,
        min_lr = parameter_config.MIN_LR,
        lr = parameter_config.LR,
        max_grad_norm=parameter_config.MAX_GRAD_NORM)  
    
    tf.set_random_seed(111)

    if hps.mode == 'train':
        batcher = batch_reader.Batcher(parameter_config.TRAIN_DIR, vocab, 'index', 
            'target', 'sentence', hps, bucketing=False, truncate_input=True)
        model = seq2seq_model.Seq2SeqModel(
            hps, vocab, num_gpus=0)
        _Train(model, batcher, parameter_config.TRAIN_STEP)
    elif hps.mode == 'eval':
        batcher = batch_reader.Batcher(parameter_config.EVALUATION_SET, vocab, 'index', 
            'target', 'sentence', hps, bucketing=False, truncate_input=True)
        model = seq2seq_model.Seq2SeqModel(
            hps, vocab, num_gpus=0)
        _Eval(model, batcher)
    elif hps.mode == 'decode':
        batcher = batch_reader.Batcher(parameter_config.DECODE_DIR, vocab, 'index', 
            'target', 'sentence', hps, bucketing=False, truncate_input=True)
        model = seq2seq_model.Seq2SeqModel(
            hps, vocab, num_gpus=0)
        if not os.path.exists(os.path.join(os.getcwd(),parameter_config.DECODE_STORE_DIR)):
            os.mkdir(os.path.join(os.getcwd(),parameter_config.DECODE_STORE_DIR))
        _Decode(model, batcher, os.path.join(parameter_config.DECODE_STORE_DIR, parameter_config.DECODE_STORE_FILE))
    elif hps.mode == 'eval_step':
        model = seq2seq_model.Seq2SeqModel(
            hps, vocab, num_gpus=0)
        _Eval_Step(model)
    else:
        print('mode_type must be train eval decode or eval_step')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'python seq2seq_model.py mode_type(train eval decode or eval_step)'
        sys.exit(-1)
    main(sys.argv[1])

