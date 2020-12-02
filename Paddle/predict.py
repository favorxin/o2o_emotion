# -*- coding:UTF-8 -*-

import os
import sys
import cv2
import logging
import argparse
import math
import numpy as np
import time
import datetime
from PIL import Image
import base64
from io import BytesIO
import string

import paddle.fluid as fluid
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

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


rec_model_dir = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/inference_large/rec/'
rec_char_dict_path = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/ppocr/utils/ppocr_keys_v1.txt'

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    #params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)

    #params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_max_side_len", type=float, default=960)

    #DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=2.0)

    #EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    #SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)
    
    #params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str, default=rec_model_dir)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=30)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--rec_char_dict_path",type=str,default=rec_char_dict_path)
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument("--enable_mkldnn", type=bool, default=False)
    parser.add_argument("--use_zero_copy_run", type=bool, default=False)
    return parser.parse_args()


def initial_logger():
    FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger


logger = initial_logger()



def create_predictor(args, mode):
    if mode == "det":
        model_dir = args.det_model_dir
    else:
        model_dir = args.rec_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/model"
    params_file_path = model_dir + "/params"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = AnalysisConfig(model_file_path, params_file_path)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
#         config.set_cpu_math_library_num_threads(6)
        if args.enable_mkldnn:
            config.enable_mkldnn()

    #config.enable_memory_optim()
    config.disable_glog_info()

    if args.use_zero_copy_run:
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
    else:
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


class TextRecognizer(object):
    def __init__(self, args):
        self.predictor, self.input_tensor, self.output_tensors = create_predictor(args, mode="rec")
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.text_len = args.max_text_length
        self.use_zero_copy_run = args.use_zero_copy_run
        char_ops_params = {
            "character_type": args.rec_char_type,
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char,
            "max_text_length": args.max_text_length
        }
        char_ops_params['loss_type'] = 'ctc'
        self.loss_type = 'ctc'
        self.char_ops = CharacterOps(char_ops_params)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
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

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        predict_time = 0
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
            if self.use_zero_copy_run:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.zero_copy_run()
            else:
                norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
                self.predictor.run([norm_img_batch])

            rec_idx_batch = self.output_tensors[0].copy_to_cpu()
            rec_idx_lod = self.output_tensors[0].lod()[0]
            predict_batch = self.output_tensors[1].copy_to_cpu()
            predict_lod = self.output_tensors[1].lod()[0]
            elapse = time.time() - starttime
            predict_time += elapse
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

args = parse_args()
text_recognizer = TextRecognizer(args)

def pred(data):
    t1 = time.time()

    img_list = [data]
    rec_res, predict_time = text_recognizer(img_list)
    chars_list = list(rec_res[0][0])
    text, score = rec_res[0][0], rec_res[0][1]
    
    t2 = time.time()
    time_consume = t2 - t1
    
    return text, score, time_consume


    
    
    
    
    
    
    
    
    