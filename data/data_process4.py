# -*- coding:utf-8 -*-
import pandas as pd
import os
import random

train_df = pd.read_csv("all_train.csv")
test_df = pd.read_csv("all_test.csv")


train_df.head()

train_df = train_df[['id', 'title', 'content', 'label']]

test_df.head()

test_df = test_df[['id', 'title', 'content']]
import re


def replace_punctuation(example):
    example = list(example)
    pre = ''
    cur = ''
    for i in range(len(example)):
        if i == 0:
            pre = example[i]
            continue
        pre = example[i - 1]
        cur = example[i]
        if re.match("[\u4e00-\u9fa5]", pre):
            if re.match("[\u4e00-\u9fa5]", cur):
                continue
            elif cur == ',':
                example[i] = '，'
            elif cur == '.':
                example[i] = '。'
            elif cur == '?':
                example[i] = '？'
            elif cur == ':':
                example[i] = '：'
            elif cur == ';':
                example[i] = '；'
            elif cur == '!':
                example[i] = '！'
            elif cur == '"':
                example[i] = '”'
            elif cur == "'":
                example[i] = "’"
    return ''.join(example)


train_df['label'] = train_df['label'].fillna(-1)
train_df = train_df[train_df['label'] != -1]
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = 0

test_df['content'] = test_df['content'].fillna('无')
train_df['content'] = train_df['content'].fillna('无')
test_df['title'] = test_df['title'].fillna('无')
train_df['title'] = train_df['title'].fillna('无')


rep_train_df = train_df[['title', 'content']].applymap(replace_punctuation)

# %%

rep_train_df = pd.concat([train_df[['id']], rep_train_df], axis=1)

# %%

rep_train_df.head()

# %%

rep_test_df = test_df[['title', 'content']].applymap(replace_punctuation)

# %%

rep_test_df = pd.concat([test_df[['id']], rep_test_df], axis=1)


train_title = []
test_title = []
test_content = []
train_content = []

r1 = "[a-zA-Z'!\"#$%&'()*+,-./:;<=>?@★[\\]^_`{|}~]+"

for train_str in rep_train_df['title']:
    train_str = re.sub(r1, '', train_str)
    train_title.append(train_str)
for train_str in rep_train_df['content']:
    train_str = re.sub(r1, '', train_str)
    train_content.append(train_str)
for test_str in rep_test_df['title']:
    test_str = re.sub(r1, '', test_str)
    test_title.append(test_str)
for test_str in rep_test_df['content']:
    test_str = re.sub(r1, '', test_str)
    test_content.append(test_str)

train_df['title'] = train_title
train_df['content'] = train_content
test_df['title'] = test_title
test_df['content'] = test_content

test_df['content'] = test_df['content'].fillna('无')
train_df['content'] = train_df['content'].fillna('无')
test_df['title'] = test_df['title'].fillna('无')
train_df['title'] = train_df['title'].fillna('无')

train_df.to_csv("replacement/train.csv", index=False)
test_df.to_csv("replacement/test.csv", index=False)

train_df = pd.read_csv("replacement/train.csv")
test_df = pd.read_csv("replacement/test.csv")

train_df = train_df[['id', 'title', 'content', 'label']]
test_df = test_df[['id', 'title', 'content']]


train_df['label'] = train_df['label'].fillna(-1)
train_df = train_df[train_df['label'] != -1]
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = 0

test_df['content'] = test_df['content'].fillna('无')
train_df['content'] = train_df['content'].fillna('无')
test_df['title'] = test_df['title'].fillna('无')
train_df['title'] = train_df['title'].fillna('无')


import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

X = np.array(train_df.index)
y = train_df.loc[:, 'label'].to_numpy()


def generate_data(random_state=42, is_pse_label=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    i = 0
    for train_index, dev_index in skf.split(X, y):
        print(i, "TRAIN:", train_index, "TEST:", dev_index)
        DATA_DIR = "./data_StratifiedKFold_{}/data_replacement_{}/".format(random_state, i)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        tmp_train_df = train_df.iloc[train_index]

        tmp_dev_df = train_df.iloc[dev_index]

        test_df.to_csv(DATA_DIR + "test.csv")
        if is_pse_label:
            pse_dir = "data_pse_{}/".format(i)
            pse_df = pd.read_csv(pse_dir + 'train.csv')

            tmp_train_df = pd.concat([tmp_train_df, pse_df], ignore_index=True, sort=False)

        tmp_train_df.to_csv(DATA_DIR + "train.csv")
        tmp_dev_df.to_csv(DATA_DIR + "dev.csv")
        print(tmp_train_df.shape, tmp_dev_df.shape)
        i += 1

generate_data(random_state=42, is_pse_label=False)