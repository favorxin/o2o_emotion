#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l

from __future__ import print_function
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from keras.layers import Input
from keras.models import Model

import pandas as pd
import os
import json
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU60%的显存
session = tf.Session(config=config)

# 基本参数
maxlen = 512
batch_size = 8
epochs = 5

# bert配置
config_path = '/data/xyang/NLP/Bert_model/tensorflow/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/xyang/NLP/Bert_model/tensorflow/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/xyang/NLP/Bert_model/tensorflow/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'


# def load_data(filename):
#     D = []
#     with open(filename, encoding='utf-8') as f:
#         for l in f:
#             title, content = l.strip().split('\t')
#             D.append((title, content))
#     return D
#
#
# # 加载数据集
# train_data = load_data('/root/csl/train.tsv')
# valid_data = load_data('/root/csl/val.tsv')
# test_data = load_data('/root/csl/test.tsv')

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
test_file = open('./round1_test_0907.json', encoding='utf-8')

train_data_all = []
for idx in range(train_df.shape[0]):
    answer_index = train_df['text'][idx].rfind(train_df['answer'][idx])
    if answer_index != -1:
        train_data_all.append((train_df['text'][idx], train_df['question'][idx], train_df['answer'][idx]))

random_order = list(range(len(train_data_all)))
np.random.shuffle(random_order)

# 划分valid
train_data = [train_data_all[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [train_data_all[j] for i, j in enumerate(random_order) if i % 10 == 0]

test_data = []

for idx in range(test_df.shape[0]):
    if test_df['answer'][idx]:
        test_data.append((test_df['text'][idx], test_df['question'][idx], test_df['answer'][idx]))

random_order = list(range(len(test_data)))

test_data = [test_data[j] for i, j in enumerate(random_order)]

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def split_str(text, answer):
    try:
        an_index = text.rfind(answer)
        text_begin = text[:an_index]
        answer_len = len(answer)
        text_end = text[an_index+answer_len:]
    except:
        answer_len = len(answer)
        if answer_len > 50:
            an_index = text.rfind(answer[answer_len//4:3*answer_len//4])
            begin_index = max(0, an_index-1-(answer_len//4))
            end_index = min(len(text), begin_index+answer_len-1)
            text_begin = text[:begin_index]
            text_end = text[end_index:]
        else:
            text_begin = text[:-answer_len-2]
            text_end = text[-2:]
    return text_begin, text_end, answer

def delete_text(cb_text, b_len, e_len, cut_len):
    if cut_len > 0:
        if e_len > cut_len:
            b_index = b_len
            an_index = len(cb_text) - e_len
            cb_text = cb_text[:-cut_len]
        elif b_len > cut_len:
            b_index = b_len - cut_len
            an_index = len(cb_text) - e_len - cut_len
            cb_text = cb_text[cut_len:]
        elif (e_len + b_len) > cut_len:
            b_index = b_len - (cut_len - e_len)
            an_index = len(cb_text) - cut_len
            cb_text = cb_text[:-e_len]
            cb_text = cb_text[cut_len-e_len:]
        else:
            b_index = b_len
            an_index = len(cb_text) - cut_len
            cb_text = cb_text[:-cut_len]
    else:
        b_index = b_len
        an_index = len(cb_text) - e_len
    return cb_text, b_index, an_index

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []
        for i in idxs:
            text, question, answer = self.data[i]
            text_begin, text_end, _ = split_str(text, answer)
            text_cut_len = max(0, len(text) - 507 + 133)
            text_combine, b_index, an_index = delete_text(text, len(text_begin), len(text_end), text_cut_len)
            text_combine_li = list(text_combine)
            text_combine_li.insert(b_index, '#')
            text_combine_li.insert(an_index + 1, '*')
            text_new = ''.join(text_combine_li)

            token_ids, segment_ids = tokenizer.encode(text_new,
                                                      question,
                                                      max_length=maxlen)
            o_token_ids = token_ids
            if np.random.random() > 0.5:
                token_ids = [
                    t if s == 0 or (s == 1 and np.random.random() > 0.3) else np.random.choice(token_ids)
                    for t, s in zip(token_ids, segment_ids)
                ]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_o_token_ids.append(o_token_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_o_token_ids = sequence_padding(batch_o_token_ids)
                yield [batch_token_ids, batch_segment_ids, batch_o_token_ids], None
                batch_token_ids, batch_segment_ids, batch_o_token_ids  = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

model.summary()

o_in = Input(shape=(None, ))
train_model = Model(model.inputs + [o_in], model.outputs + [o_in])

# 交叉熵作为loss，并mask掉输入部分的预测
y_true = train_model.input[2][:, 1:]  # 目标tokens
y_mask = train_model.input[1][:, 1:]
y_pred = train_model.output[0][:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

train_model.add_loss(cross_entropy)
train_model.compile(optimizer=Adam(1e-5))


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, answer, topk=1):
        max_c_len = maxlen - self.maxlen
        text_begin, text_end, _ = split_str(text, answer)
        text_cut_len = max(0, len(text) - 507 + 133)
        text_combine, b_index, an_index = delete_text(text, len(text_begin), len(text_end), text_cut_len)
        text_combine_li = list(text_combine)
        text_combine_li.insert(b_index, '#')
        text_combine_li.insert(an_index + 1, '*')
        text_new = "".join(text_combine_li)

        token_ids, segment_ids = tokenizer.encode(text_new, max_length=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None,
                      end_id=tokenizer._token_sep_id,
                      maxlen=133)

def predict_to_file(test_file, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    test_data = json.load(test_file)
    # with open(filename, 'w', encoding='utf-8') as f:
    for param in tqdm(iter(test_data), desc=u'正在预测(共%s条样本)' % len(test_data)):
        for idx in range(len(param['annotations'])):
            q = autotitle.generate(param['text'],param['annotations'][idx]['A'])
            param['annotations'][idx]['Q'] = q
            # s = '%s\t%s\t%s\n' % (q, d[2], d[0])
    with open(filename, "w") as f:
        json.dump(test_data, f)
    print("保存文件完成...")

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('./best_model.baseline_e5.weights')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for text, question, answer in tqdm(data):
            total += 1
            question = ' '.join(question).lower()
            pred_question = ' '.join(autotitle.generate(text, answer, topk)).lower()
            if pred_question.strip():
                scores = self.rouge.get_scores(hyps=pred_question, refs=question)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(references=[question.split(' ')],
                                      hypothesis=pred_question.split(' '),
                                      smoothing_function=self.smooth)
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    train_model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])

else:
    model.load_weights('./best_model.baseline_e5.weights')
    predict_to_file(test_file, "./inference.json")