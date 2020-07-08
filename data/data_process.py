# -*- coding:utf-8 -*-
import pandas as pd

chusai_train = pd.read_csv("./Train_DataSet.csv")
chusai_train_label = pd.read_csv("./Train_DataSet_Label.csv")
chusai_train=chusai_train.merge(chusai_train_label,on='id',how='left')
chusai_train['label']=chusai_train['label'].fillna(-1)
chusai_train=chusai_train[chusai_train['label']!=-1]
chusai_train['label']=chusai_train['label'].astype(int)
chusai_train["content"] = chusai_train["content"].fillna("无")
chusai_train["title"] = chusai_train["title"].fillna("无")
chusai_train.head()

fusai_train = pd.read_csv("./Second_DataSet.csv", header=None)
fusai_train_label = pd.read_csv("./Second_DataSet_Label.csv", header=None)
fusai_train.columns = ['id', 'title', 'content']
fusai_train_label.columns = ['id', 'label']
fusai_train=fusai_train.merge(fusai_train_label,on='id',how='left')
fusai_train['label']=fusai_train['label'].fillna(-1)
fusai_train=fusai_train[fusai_train['label']!=-1]
fusai_train['label']=fusai_train['label'].astype(int)
fusai_train["content"] = fusai_train["content"].fillna("无")
fusai_train["title"] = fusai_train["title"].fillna("无")
fusai_train.head()

train_df = chusai_train.append(fusai_train)
print(train_df.shape)
train_df.to_csv("./all_train.csv")

fusai_test = pd.read_csv("./Second_TestDataSet.csv", header=None)
fusai_test.columns = ['id', 'title', 'content']
fusai_test["content"] = fusai_test["content"].fillna("无")
fusai_test["title"] = fusai_test["title"].fillna("无")
fusai_test.to_csv("all_test.csv")
