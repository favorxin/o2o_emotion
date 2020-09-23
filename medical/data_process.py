#!/usr/bin/env python
# -*- coding:utf-8 -*-

count_cut_1 = 0
count_cut_2 = 0
for i in range(len(train_data)):
    for j in range(len(train_data[i]['annotations'])):
        text_new = replace_text_answer(train_data[i]['text'], train_data[i]['annotations'][j]['A'])
        if len(text_new) > 378:
            text_cut = text_new[:-(len(text_new)-378)]
            if "#" not in text_cut:
                count_cut_1 += 1
            if "*" not in text_cut:
                count_cut_2 += 1
print(count_cut_1, count_cut_2)
