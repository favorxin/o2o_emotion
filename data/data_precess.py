# -*- coding: utf-8 -*-
import pandas as pd

gupiao_df = pd.read_excel('market.xlsx', sheet_name=0)
# print("gupiao: ",gupiao_df.head(5))
jijin_df = pd.read_excel('market.xlsx', sheet_name=1)
# print("jijin: ",jijin_df.head(5))
qihuo_df = pd.read_excel('market.xlsx', sheet_name=2)
# print("qihuo: ",qihuo_df.head(5))
waihui_df = pd.read_excel('market.xlsx', sheet_name=3)
# print("waihui: ",waihui_df.head(5))
gupiao_df['label'] = 0
jijin_df['label'] = 1
qihuo_df['label'] = 2
waihui_df['label'] = 3

gupiao_df = gupiao_df[['id', 'title', 'label']]
jijin_df = jijin_df[['id', 'title', 'label']]
qihuo_df = qihuo_df[['id', 'title', 'label']]
waihui_df = waihui_df[['id', 'title', 'label']]
print(gupiao_df.shape, jijin_df.shape, qihuo_df.shape, waihui_df.shape)

gupiao_df['title'] = gupiao_df['title'].fillna('无')
jijin_df['title'] = jijin_df['title'].fillna('无')
qihuo_df['title'] = qihuo_df['title'].fillna('无')
waihui_df['title'] = waihui_df['title'].fillna('无')

gupiao_repeat = gupiao_df[gupiao_df.duplicated(['id', 'title'])]
print("gupiao_repeat",gupiao_repeat.shape)
print("gupiao zong",gupiao_df.shape)
gupiao_rm = gupiao_df[-gupiao_df.duplicated(['id', 'title'])]
print("gupiao rm",gupiao_rm.shape)

jijin_repeat = jijin_df[jijin_df.duplicated(['id', 'title'])]
print("jijin repeat", jijin_repeat.shape)
print("jijin zong",jijin_df.shape)
jijin_rm = jijin_df[-jijin_df.duplicated(['id', 'title'])]
print("jijin rm",jijin_rm.shape)

qihuo_repeat = qihuo_df[qihuo_df.duplicated(['id', 'title'])]
print("qihuo repeat", qihuo_repeat.shape)
print("qihuo zong",qihuo_df.shape)
qihuo_rm = qihuo_df[-qihuo_df.duplicated(['id', 'title'])]
print("qihuo rm",qihuo_rm.shape)

waihui_repeat = waihui_df[waihui_df.duplicated(['id', 'title'])]
print("waihui repeat", waihui_repeat.shape)
print("waihui zong",waihui_df.shape)
waihui_rm = waihui_df[-waihui_df.duplicated(['id', 'title'])]
print("waihui rm",waihui_rm.shape)

df_concat = pd.merge(gupiao_rm, jijin_rm, how='outer')
df_concat = pd.merge(df_concat, qihuo_rm, how='outer')
df_concat = pd.merge(df_concat, waihui_rm, how='outer')
print("lianjie zong",df_concat.shape)
df_concat_repeat = df_concat[df_concat.duplicated(['id', 'title'])]
print("lianjie repeat",df_concat_repeat.shape)


# data1 = gupiao_df.drop_duplicates()

# df_concat = pd.merge(gupiao_df, jijin_df, how='outer')
# df_concat = pd.merge(df_concat, qihuo_df, how='outer')
# df_concat = pd.merge(df_concat, waihui_df, how='outer')
# print(df_concat.shape)
# print(df_concat.iloc[1])

