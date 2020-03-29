# ==============================================================================
# -*- coding: utf-8 -*-
# @Time    : 2020-03-08 PM 6:16
# @Author  : deymm
# @Email   : deymm671@126.com
# ==============================================================================

import numpy as np
from sklearn import metrics
from scipy.stats import ks_2samp

def calculate_acc(y_true, y_predprob, threshold=0.5):
	"""计算Accuracy

	Args:
      y_true: list, 真实标签
      y_predprob: list, 预测值
      threshold: int, 正负类阈值

    Returns:
       Accuracy
    """
	y_pred = []
	for value in y_predprob:
		if value > threshold:
			y_pred.append(1)
		else:
			y_pred.append(0)
	acc = metrics.accuracy_score(y_true, y_pred)
	auc = metrics.roc_auc_score(y_true, y_predprob)

	return acc

def calculate_auc(y_true, y_predprob):
	"""计算AUC

	Args:
      y_true: list, 真实标签
      y_predprob: list, 预测值

    Returns:
       AUC
    """
	auc = metrics.roc_auc_score(y_true, y_predprob)
	return auc


def calculate_ks(y_true, y_predprob):
    """计算KS

	Args:
      y_true: list, 真实标签
      y_predprob: list, 预测值

    Returns:
       KS
    """  
    get_ks = lambda y_predprob, y_true: ks_2samp(y_predprob[y_true == 1], y_predprob[y_true == 0]).statistic
    ks = get_ks(np.array(y_predprob), np.array(y_true))

    return ks

