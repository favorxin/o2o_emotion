# -*-coding: utf-8 -*-

import pandas as pd
import os
import random

chusai_train = pd.read_csv("./data/Train_DataSet.csv")
chusai_label = pd.read_csv("./data/Train_DataSet_Label.csv")
chusai_train = chusai_train.merge(chusai_label, on='id', how='left')
chusai_train['label']=chusai_train['label'].fillna(-1)
chusai_train=chusai_train[chusai_train['label']!=-1]
chusai_train['label']=chusai_train['label'].astype(int)
chusai_train["content"]=chusai_train["content"].fillna("wu")
chusai_train["title"]=chusai_train["title"].fillna("wu")
chusai_train.head()

print(chusai_train.isnull().sum())

chusai_train.to_csv("./data/all_train.csv")
