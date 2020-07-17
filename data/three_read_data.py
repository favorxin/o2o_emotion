# -*- coding: utf-8 -*-
import pandas as pd

all_data = pd.read_csv('test.csv')
print(all_data.shape)

all_data = all_data[['id', 'title', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6']]
print(all_data.shape)

print(all_data['label_0'].value_counts())
