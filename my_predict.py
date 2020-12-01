import os
import sys
import cv2
import copy
import numpy as np
import math
import time
import imghdr
import logging
import paddle.fluid as fluid
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
import string

# from ppocr.utils.character import CharacterOps

def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists

def check_and_read_gif(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logging.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False

def create_predictor_rec(mode):
    """
    create predictor for inference
    :param args: params for prediction engine
    :param mode: mode
    :return: predictor
    """

    model_dir = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/inference_large/rec/'
    model_file_path = model_dir + "/model"
    params_file_path = model_dir + "/params"

    config = AnalysisConfig(model_file_path, params_file_path)

    config.enable_use_gpu(8000, 0)

    # config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(True)

    predictor = create_paddle_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_tensor(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_tensor(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors

class CharacterOps(object):
    """
    Convert between text-label and text-index
    Args:
        config: config from yaml file
    """

    def __init__(self, config):
        self.character_type = config['character_type']
        self.loss_type = config['loss_type']
        self.max_text_len = config['max_text_length']
        # use the default dictionary(36 char)
        if self.character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        # use the custom dictionary
        elif self.character_type in [
                "ch", 'japan', 'korean', 'french', 'german'
        ]:
            character_dict_path = config['character_dict_path']
            add_space = False
            if 'use_space_char' in config:
                add_space = config['use_space_char']
            self.character_str = ""
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if add_space:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)
        self.beg_str = "sos"
        self.end_str = "eos"
        # add start and end str for attention
        if self.loss_type == "attention":
            dict_character = [self.beg_str, self.end_str] + dict_character
        elif self.loss_type == "srn":
            dict_character = dict_character + [self.beg_str, self.end_str]
        # create char dict
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def encode(self, text):
        """
        convert text-label into text-index.
        Args:
            text: text labels of each image. [batch_size]
        Return:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
        """
        # Ignore capital
        if self.character_type == "en":
            text = text.lower()

        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        text = np.array(text_list)
        return text

    def decode(self, text_index, is_remove_duplicate=False):
        """
        convert text-index into text-label.
        Args:
            text_index: text index for each image
            is_remove_duplicate: Whether to remove duplicate characters,
                                 The default is False
        Return:
            text: text label
        """
        char_list = []
        char_num = self.get_char_num()

        if self.loss_type == "attention":
            beg_idx = self.get_beg_end_flag_idx("beg")
            end_idx = self.get_beg_end_flag_idx("end")
            ignored_tokens = [beg_idx, end_idx]
        else:
            ignored_tokens = [char_num]

        for idx in range(len(text_index)):
            if text_index[idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                if idx > 0 and text_index[idx - 1] == text_index[idx]:
                    continue
            char_list.append(self.character[int(text_index[idx])])
        text = ''.join(char_list)
        return text

    def get_char_num(self):
        """
        Get character num
        """
        return len(self.character)

class TextRecognizer(object):
    def __init__(self):
        self.predictor, self.input_tensor, self.output_tensors =\
            create_predictor_rec(mode="rec")
        self.use_zero_copy_run = False
        self.rec_image_shape = [3, 32, 320]
        self.character_type = 'ch'
        self.rec_batch_num = 6
        self.rec_algorithm = 'CRNN'
        self.text_len = 25
        char_ops_params = {
            "character_type": 'ch',
            "character_dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
            "use_space_char": True,
            "max_text_length": 25
        }
        char_ops_params['loss_type'] = 'ctc'
        self.loss_type = 'ctc'
        self.char_ops = CharacterOps(char_ops_params)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = 3, 32, 320
        assert imgC == img.shape[2]
        wh_ratio = max(max_wh_ratio, imgW * 1.0 / imgH)
        if self.character_type == "ch":
            imgW = int((32 * wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        #rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        predict_time = 0
        start_time = time.time()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch, axis=0)
            norm_img_batch = norm_img_batch.copy()

            starttime = time.time()
            norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
            self.predictor.run([norm_img_batch])

            rec_idx_batch = self.output_tensors[0].copy_to_cpu()
            rec_idx_lod = self.output_tensors[0].lod()[0]
            predict_batch = self.output_tensors[1].copy_to_cpu()
            predict_lod = self.output_tensors[1].lod()[0]
            elapse = time.time() - starttime
            predict_time += elapse
            start_time_in = time.time()
            for rno in range(len(rec_idx_lod) - 1):
                beg = rec_idx_lod[rno]
                end = rec_idx_lod[rno + 1]
                rec_idx_tmp = rec_idx_batch[beg:end, 0]
                preds_text = self.char_ops.decode(rec_idx_tmp)
                beg = predict_lod[rno]
                end = predict_lod[rno + 1]
                probs = predict_batch[beg:end, :]
                ind = np.argmax(probs, axis=1)
                blank = probs.shape[1]
                valid_ind = np.where(ind != (blank - 1))[0]
                if len(valid_ind) == 0:
                    continue
                score = np.mean(probs[valid_ind, ind[valid_ind]])
                # rec_res.append([preds_text, score])
                rec_res[indices[beg_img_no + rno]] = [preds_text, score]

        return rec_res, predict_time

def main(image_dir):
    start_time = time.time()
    image_file_list = get_image_file_list(image_dir)
    get_file_time = time.time()
    text_recognizer = TextRecognizer()
    valid_image_file_list = []
    img_list = []

    # if os.path.isdir(image_dir):
    #     images_file = os.listdir(image_dir)
    #     for file in images_file:
    #         name, ext = os.path.splitext(file)
    #         if ext in ['.jpg', '.bmp', '.png', '.jpeg', '.rgb', '.tif', '.tiff']:
    #             img_file = os.path.join(image_dir, file)
    #             img = cv2.imread(img_file)
    #         valid_image_file_list.append(img_file)
    #         img_list.append(img)
    # elif os.path.isfile(image_dir):
    #     name, ext = os.path.splitext(image_dir.split('/')[-1])
    #     if ext in ['.jpg', '.bmp', '.png', '.jpeg', '.rgb', '.tif', '.tiff']:
    #         img = cv2.imread(image_dir)
    #     valid_image_file_list.append(image_dir)
    #     img_list.append(img)

    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    load_img_list_time = time.time()

    try:
        rec_res, predict_time = text_recognizer(img_list)
    except Exception as e:
        print(e)
        exit()
    end_time = time.time()
    print("get file time: {}, load img time: {}, recognize time: {}".format( \
        get_file_time-start_time, load_img_list_time-get_file_time, end_time-load_img_list_time))
    for ino in range(len(img_list)):
        print("Predicts of %s:%s" % (valid_image_file_list[ino], rec_res[ino]))
    print("Total predict time for %d images:%.3f" %
          (len(img_list), predict_time))


if __name__ == "__main__":
    image_dir = 'D:/CV/main_job/data_job/wrong_recognition_pic/t1.bmp'
    main(image_dir)
