# -*- coding: utf-8 -*-
import pandas as pd

train_df = pd.read_excel('0717_masket.xlsx')
test_df = pd.read_excel('0717_masket_2.xlsx')
print(train_df.shape)
print(test_df.shape)
train_df = pd.merge(train_df, test_df, how='outer')
print(train_df.shape)
train_df = train_df[['id', 'title', 'label']]
train_df['label'] = train_df['label'].astype(int)
gushi_df = train_df[train_df['label']==0]
gushi_df['label'] = 7
print(gushi_df['label'].value_counts())
jijin_df = train_df[train_df['label']==1]
qihuo_df = train_df[train_df['label']==2]
zhaijuan_df = train_df[train_df['label']==3]
waihui_df = train_df[train_df['label']==4]
licai_df = train_df[train_df['label']==5]
qiquan_df = train_df[train_df['label']==6]
print('gushi: {}, jijin: {}, qihuo: {}, zhaiquan: {}, waihui: {}, licai: {}, qiquan: {}'.format(
    gushi_df.shape, jijin_df.shape, qihuo_df.shape, zhaijuan_df.shape, waihui_df.shape, licai_df.shape, qiquan_df.shape
))
gushi_repeat = gushi_df[gushi_df.duplicated(['id', 'title'])]
print("gushi repeat: ",gushi_repeat.shape)
gushi_rm = gushi_df[-gushi_df.duplicated(['id', 'title'])]
print("gushi rm", gushi_rm.shape)

jijin_repeat = jijin_df[jijin_df.duplicated(['id', 'title'])]
print("jijin repeat: ",jijin_repeat.shape)
jijin_rm = jijin_df[-jijin_df.duplicated(['id', 'title'])]
print("jijin rm", jijin_rm.shape)

qihuo_repeat = qihuo_df[qihuo_df.duplicated(['id', 'title'])]
print("qihuo repeat: ",qihuo_repeat.shape)
qihuo_rm = qihuo_df[-qihuo_df.duplicated(['id', 'title'])]
print("qihuo rm", qihuo_rm.shape)

zhaijuan_repeat = zhaijuan_df[zhaijuan_df.duplicated(['id', 'title'])]
print("zhaijuan repeat: ",zhaijuan_repeat.shape)
zhaijuan_rm = zhaijuan_df[-zhaijuan_df.duplicated(['id', 'title'])]
print("zhaijuan rm", zhaijuan_rm.shape)

waihui_repeat = waihui_df[waihui_df.duplicated(['id', 'title'])]
print("waihui repeat: ",waihui_repeat.shape)
waihui_rm = waihui_df[-waihui_df.duplicated(['id', 'title'])]
print("waihui rm", waihui_rm.shape)

licai_repeat = licai_df[licai_df.duplicated(['id', 'title'])]
print("licai repeat: ",licai_repeat.shape)
licai_rm = licai_df[-licai_df.duplicated(['id', 'title'])]
print("licai rm", licai_rm.shape)

qiquan_repeat = qiquan_df[qiquan_df.duplicated(['id', 'title'])]
print("qiquan repeat: ",qiquan_repeat.shape)
qiquan_rm = qiquan_df[-qiquan_df.duplicated(['id', 'title'])]
print("qiquan rm", qiquan_rm.shape)

df_concat = pd.merge(gushi_rm, jijin_rm, how='outer')
df_concat = pd.merge(df_concat, qihuo_rm, how='outer')
df_concat = pd.merge(df_concat, zhaijuan_rm, how='outer')
df_concat = pd.merge(df_concat, waihui_rm, how='outer')
df_concat = pd.merge(df_concat, licai_rm, how='outer')
df_concat = pd.merge(df_concat, qiquan_rm, how='outer')
print(df_concat.shape)
print(df_concat['label'].value_counts())

# total_df = pd.read_csv('test.csv')
# # for file in [jijin_rm, qihuo_rm, zhaijuan_rm, waihui_rm, licai_rm, qiquan_rm]:
# file = qiquan_rm
# print("start!!!")
# df_concat = pd.merge(total_df, file, how='outer')
# same_data = df_concat[df_concat.duplicated(['id', 'title'])]
# print("repeat: {}".format(same_data.shape))
# df_merge = pd.merge(gushi_rm, file, how='right', on = ['id', 'title'])
# now_df = df_concat[-df_concat.duplicated(['id', 'title'])]
# total_df = pd.merge(now_df, df_merge, how='outer')
# print("chongfu: ", df_merge.shape)
# print("total: ",total_df.shape)
# print(total_df.head(0))
# total_df['label'] = total_df['label'].fillna(0)
# # total_df['label_y'] = total_df['label_y'].fillna(0)
# # total_df['label_x'] = total_df['label_x'].astype(int)
# # total_df['label_y'] = total_df['label_y'].astype(int)
# #
# total_df = total_df[['id', 'title', 'label_x', 'label_y', 'label_1', 'label_2', 'label_3', 'label_4', 'label']]
# # #, 'label_1','label_2','label_3','label_4','label']]
# total_df.to_csv('test.csv')
# print("ending!!!")
