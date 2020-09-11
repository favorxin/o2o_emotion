#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import pandas as pd
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import text_segmentate
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 占用GPU60%的显存

session = tf.Session(config=config)

train_file = open('./round1_train_0907.json', encoding='utf-8')
test_file = open('./round1_test_0907.json', encoding='utf-8')

train_data = json.load(train_file)
test_data = json.load(test_file)

all_data_list = []
for i in range(len(train_data)):
    for j in range(len(train_data[i]['annotations'])):
        dict_cur = {}
        dict_cur['id'] = train_data[i]['id']
        dict_cur['text'] = train_data[i]['text']
        dict_cur['question'] = train_data[i]['annotations'][j]['Q']
        dict_cur['answer'] = train_data[i]['annotations'][j]['A']
        all_data_list.append(dict_cur)

id_list = []
text_list = []
question_list = []
answer_list = []
for i in range(len(all_data_list)):
    id_list.append(all_data_list[i]['id'])
    text_list.append(all_data_list[i]['text'])
    question_list.append(all_data_list[i]['question'])
    answer_list.append(all_data_list[i]['answer'])

train_data = pd.DataFrame(columns = ["id", "text", "question", "answer"])
train_data['id'] = id_list
train_data['text'] = text_list
train_data['question'] = question_list
train_data['answer'] = answer_list
print(train_data.shape)

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

# 基本参数
max_p_len = 432
max_q_len = 16
max_a_len = 64
batch_size = 4
epochs = 40

# bert配置
config_path = '/data/xyang/NLP/Bert_model/tf/chinese_roberta_wwm_ext/bert_config.json'
checkpoint_path = '/data/xyang/NLP/Bert_model/tf/chinese_roberta_wwm_ext/bert_model.ckpt'
dict_path = '/data/xyang/NLP/Bert_model/tf/chinese_roberta_wwm_ext/vocab.txt'

# 筛选数据
seps, strips = u'\n。！？!?；;，, ', u'；;，, '
data = []

for idx in range(train_data.shape[0]):
    if train_data['answer'][idx]:
        for t in text_segmentate(train_data['text'][idx], max_p_len - 2, seps, strips):
            if train_data['answer'][idx] in t:
                data.append((t, train_data['question'][idx], train_data['answer'][idx]))

random_order = list(range(len(data)))
np.random.shuffle(random_order)
json.dump(random_order, open('../random_order.json', 'w'), indent=4)

# 划分valid
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (p, q, a) in self.sample(random):
            p_token_ids, _ = tokenizer.encode(p, maxlen=max_p_len + 1)
            a_token_ids, _ = tokenizer.encode(a, maxlen=max_a_len)
            q_token_ids, _ = tokenizer.encode(q, maxlen=max_q_len)
            token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * len(p_token_ids)
            segment_ids += [1] * (len(token_ids) - len(p_token_ids))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()

class QuestionAnswerGeneration(AutoRegressiveDecoder):
    """随机生成答案，并且通过beam search来生成问题
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, passage, topk=5):
        token_ids, segment_ids = tokenizer.encode(passage, maxlen=max_p_len)
        a_ids = self.random_sample([token_ids, segment_ids], 1,
                                   topk)[0]  # 基于随机采样
        token_ids += list(a_ids)
        segment_ids += [1] * len(a_ids)
        q_ids = self.beam_search([token_ids, segment_ids],
                                 topk)  # 基于beam search
        return (tokenizer.decode(q_ids), tokenizer.decode(a_ids))

qag = QuestionAnswerGeneration(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=max_q_len
)

def predict_to_file(data, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q, a = qag.generate(d[0])
            s = '%s\t%s\t%s\n' % (q, a, d[0])
            f.write(s)
            f.flush()

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')

if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=1000,
        epochs=epochs,
        callbacks=[evaluator]
    )
else:
    model.load_weights('./best_model.weights')
    # predict_to_file(valid_data, 'qa.csv')

