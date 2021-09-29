"""
https://www.kaggle.com/sudalairajkumar/simple-leaky-exploration-notebook-quora?scriptVersionId=1184830
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost

from kaggle.quora_question_pairs.load_data import LoadData
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


# 加载训练 测试 数据集
train_df = LoadData().load_train()
test_df = LoadData().load_test()

print(train_df.shape)
print(test_df.shape)

print(train_df.head())
print(test_df.head())






