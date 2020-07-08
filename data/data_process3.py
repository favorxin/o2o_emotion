# -*- coding:utf-8 -*-
import pandas as pd
import os
import random

train_df=pd.read_csv("all_train.csv")
test_df=pd.read_csv("all_test.csv")

train_df = train_df[['id', 'title', 'content', 'label']]
test_df = test_df[['id', 'title', 'content']]

train_df['label']=train_df['label'].fillna(-1)
train_df=train_df[train_df['label']!=-1]
train_df['label']=train_df['label'].astype(int)
test_df['label']=0

test_df['content']=test_df['content'].fillna('无')
train_df['content']=train_df['content'].fillna('无')
test_df['title']=test_df['title'].fillna('无')
train_df['title']=train_df['title'].fillna('无')

import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

X = np.array(train_df.index)
y = train_df.loc[:,'label'].to_numpy()


def generate_data(random_state=24, is_pse_label=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    i = 0
    for train_index, dev_index in skf.split(X, y):
        print(i, "TRAIN:", train_index, "TEST:", dev_index)
        DATA_DIR = "./data_StratifiedKFold_{}/data_origin_{}/".format(random_state, i)
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


generate_data(random_state=24, is_pse_label=False)
