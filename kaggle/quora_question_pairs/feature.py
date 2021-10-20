"""
https://www.kaggle.com/sudalairajkumar/simple-leaky-exploration-notebook-quora?scriptVersionId=1184830
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost

from collections import defaultdict
from kaggle.quora_question_pairs.load_data import LoadData
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

eng_stopwords = set(stopwords.words('english'))
color = sns.color_palette()

# 加载训练 测试 数据集
train_df = LoadData().load_train()
test_df = LoadData().load_test()

print(train_df.shape)
print(test_df.shape)

# print(train_df.head())
# print(test_df.head())


# label distribution
is_dup = train_df['is_duplicate'].value_counts()
# plt.figure(figsize=(8, 4))
# sns.barplot(is_dup.index, is_dup.values, alpha=0.8, color=color[0])
# plt.xlabel('Is Duplicate', fontsize=12)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.show()

label_dis = is_dup / is_dup.sum()
print(label_dis)

# words distribution
all_ques_df = pd.DataFrame(pd.concat([train_df['question1'], train_df['question2']]))
all_ques_df.columns = ['questions']
all_ques_df['num_of_words'] = all_ques_df['questions'].apply(lambda x: len((str(x).split())))
print(all_ques_df.head())

cnt_srs = all_ques_df['num_of_words'].value_counts()


# print(cnt_srs)

# plt.figure(figsize=(12, 6))
# sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
# plt.xlabel('Number of words in the question', fontsize=12)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xticks(rotation='vertical')
# plt.show()

# characters distribution
# all_ques_df['num_of_chars'] = all_ques_df['questions'].apply(lambda x: len(str(x)))
# cnt_srs = all_ques_df['num_of_chars'].value_counts()
#
#
# # plt.figure(figsize=(50, 8))
# # sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
# # plt.xlabel('Number of chars in the question', fontsize=12)
# # plt.ylabel('Number of Occurrences', fontsize=12)
# # plt.xticks(rotation='vertical')
# # plt.show()


def get_unigrams(que):
    return [word for word in word_tokenize(que.lower()) if word not in eng_stopwords]
    pass


def get_common_unigrams(row):
    return len(set(row['unigrams_ques1']).intersection(set(row['unigrams_ques2'])))
    pass


def get_common_unigram_ratio(row):
    return float(row['unigrams_common_count']) / max(len(set(row['unigrams_ques1']).union(set(row['unigrams_ques2']))),
                                                     1)
    pass


train_df['unigrams_ques1'] = train_df['question1'].apply(lambda x: get_unigrams(str(x)))
train_df['unigrams_ques2'] = train_df['question2'].apply(lambda x: get_unigrams(str(x)))
train_df['unigrams_common_count'] = train_df.apply(lambda row: get_common_unigrams(row), axis=1)
train_df['unigrams_common_ratio'] = train_df.apply(lambda row: get_common_unigram_ratio(row), axis=1)

cnt_srs = train_df['unigrams_common_count'].value_counts()

# 查看相同词数量分布
# plt.figure(figsize=(12, 6))
# sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
# plt.xlabel('Common unigrams count', fontsize=12)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xticks(rotation='vertical')
# plt.show()

# 箱形图
# plt.figure(figsize=(12, 6))
# plt.subplot(211)
# sns.boxplot(x='is_duplicate', y='unigrams_common_count', data=train_df)
# plt.xlabel('Is duplicate', fontsize=12)
# plt.ylabel('Common unigrams count', fontsize=12)
# plt.title('unigrams_common_count')
#
# plt.subplot(212)
# # plt.figure(figsize=(12, 6))
# sns.boxplot(x='is_duplicate', y='unigrams_common_ratio', data=train_df)
# plt.xlabel('Is duplicate', fontsize=12)
# plt.ylabel('Common unigrams ratio', fontsize=12)
# plt.title('unigrams_common_ratio')
# plt.show()

#
ques = pd.concat([
    train_df[['question1', 'question2']],
    test_df[['question1', 'question2']]
],
    axis=0
).reset_index(drop='index')
print(ques.shape)

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])
    pass


def q1_freq(row):
    return (len(q_dict[row['question1']]))
    pass


def q2_freq(row):
    return (len(q_dict[row['question2']]))
    pass


def q1_q2_intersect(row):
    return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))
    pass


train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)

print()
