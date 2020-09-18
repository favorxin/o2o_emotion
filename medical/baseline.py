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
import os
import json
import pandas as pd
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用GPU60%的显存
session = tf.Session(config=config)

# 基本参数
max_text_len = 384
max_question_len = 64
max_answer_len = 64
batch_size = 5
epochs = 5

# bert配置
config_path = '/data/xyang/NLP/Bert_model/tensorflow/chinese_roberta_wwm_ext/bert_config.json'
checkpoint_path = '/data/xyang/NLP/Bert_model/tensorflow/chinese_roberta_wwm_ext/bert_model.ckpt'
dict_path = '/data/xyang/NLP/Bert_model/tensorflow/chinese_roberta_wwm_ext/vocab.txt'


# def load_data(filename):
#     D = []
#     with open(filename, encoding='utf-8') as f:
#         for l in f:
#             title, content = l.strip().split('\t')
#             D.append((title, content))
#     return D
#
# # 加载数据集
# train_data = load_data('/root/csl/train.tsv')
# valid_data = load_data('/root/csl/val.tsv')
# test_data = load_data('/root/csl/test.tsv')

### 加载数据集
train_file = open('./round1_train_0907.json', encoding='utf-8')
test_file = open('./round1_test_0907.json', encoding='utf-8')

train_or_data = json.load(train_file)
test_or_data = json.load(test_file)

train_df = pd.read_csv('./train.csv', encoding='utf-8')
test_df = pd.read_csv('./test.csv', encoding='utf-8')

def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]

# 筛选数据
seps, strips = u'\n。！？!?；;，, ', u'；;，, '
data = []

for idx in range(train_df.shape[0]):
    if train_df['answer'][idx]:
        for t in text_segmentate(train_df['text'][idx], max_text_len - 2, seps, strips):
            if train_df['answer'][idx] in t:
                data.append((t, train_df['question'][idx], train_df['answer'][idx]))

random_order = list(range(len(data)))
np.random.shuffle(random_order)
json.dump(random_order, open('../random_order.json', 'w'), indent=4)

# 划分valid
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

test_data = []

for idx in range(test_df.shape[0]):
    if test_df['answer'][idx]:
        for t in text_segmentate(test_df['text'][idx], max_text_len - 2, seps, strips):
            if test_df['answer'][idx] in t:
                data.append((t, test_df['question'][idx], test_df['answer'][idx]))

random_order = list(range(len(data)))

test_data = [data[j] for i, j in enumerate(random_order)]

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

#
# class data_generator(DataGenerator):
#     """数据生成器
#     """
#     def __iter__(self, random=False):
#         idxs = list(range(len(self.data)))
#         if random:
#             np.random.shuffle(idxs)
#         batch_token_ids, batch_segment_ids = [], []
#         for i in idxs:
#             title, content = self.data[i]
#             token_ids, segment_ids = tokenizer.encode(content,
#                                                       title,
#                                                       max_length=maxlen)
#             batch_token_ids.append(token_ids)
#             batch_segment_ids.append(segment_ids)
#             if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
#                 batch_token_ids = sequence_padding(batch_token_ids)
#                 batch_segment_ids = sequence_padding(batch_segment_ids)
#                 yield [batch_token_ids, batch_segment_ids], None
#                 batch_token_ids, batch_segment_ids = [], []

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
        """
        idxs = list(range(len(self.data)))
        batch_token_ids, batch_segment_ids = [], []
        for i in idxs:
            p, q, a = self.data[i]
            p_token_ids, _ = tokenizer.encode(p, max_length=max_text_len + 1)
            a_token_ids, _ = tokenizer.encode(a, max_length=max_answer_len)
            q_token_ids, _ = tokenizer.encode(q, max_length=max_question_len)
            token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * len(p_token_ids + a_token_ids[1:])
            segment_ids += [1] * (len(q_token_ids[1:]))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or i==idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

model.summary()

# 交叉熵作为loss，并mask掉输入部分的预测
y_true = model.input[0][:, 1:]  # 目标tokens
y_mask = model.input[1][:, 1:]
y_pred = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


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
        # max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, max_length=max_text_len)
        answer_token_ids, answer_segment_ids = tokenizer.encode(answer, max_length=max_answer_len)
        token_ids += list(answer_token_ids[1:])
        segment_ids += [0] * len(answer_token_ids[1:])

        output_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None,
                      end_id=tokenizer._token_sep_id,
                      maxlen=64)

def predict_to_file(data, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q = autotitle.generate(d[0],d[2])
            s = '%s\t%s\t%s\n' % (q, d[2], d[0])
            f.write(s)
            f.flush()

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('./best_model.baseline.weights')  # 保存模型
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
    # len(train_generator)
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])

else:
    model.load_weights('./best_model.baseline.weights')
    predict_to_file(test_data, 'new_qa.csv')