import os
import sys
import logging
import argparse

import cv2
import copy
import numpy as np
import math
import time
import imghdr
import pandas as pd
from PIL import Image
from utility import draw_ocr
from utility import draw_ocr_box_txt

from predict import pred
import detection as detec

det_model_dir = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/inference/det/'
rec_model_dir = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/inference/rec/'
rec_char_dict_path = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/ppocr/utils/ppocr_keys_v1.txt'
vis_font_path = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/doc/simfang.ttf'

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)

    # params for text detector
    parser.add_argument("--image_dir", type=str, default='D:/CV/main_job/bangdan_examples/need_recognition/test_pic/')
    # 'D:/CV/main_job/EDB_nianjian/test_pic/'  'D:/CV/main_job/bangdan_examples/need_recognition/test_pic/'
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default=det_model_dir)
    parser.add_argument("--det_max_side_len", type=float, default=960)

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=2.0)

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str, default=rec_model_dir)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=30)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--rec_char_dict_path", type=str, default=rec_char_dict_path)
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument("--enable_mkldnn", type=bool, default=False)
    parser.add_argument("--use_zero_copy_run", type=bool, default=False)
    parser.add_argument(
        "--vis_font_path", type=str, default=vis_font_path)
    return parser.parse_args()

def initial_logger():
    FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger

logger = initial_logger()
args = parse_args()

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

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = detec.Detecting(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        # rec_res, elapse = self.text_recognizer(img_crop_list)
        rec_res = []
        elapse = 0
        for img in img_crop_list:
            txt, score, time_u = pred(img)
            rec_res.append([txt, score])
            elapse += time_u
        print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def xy_info(boxes, results):
    data_list = []
    for idx, box in enumerate(boxes):
        info = results[idx]
        box_x, box_y = box[0]
        data_list.append([info, int(box_x), int(box_y)])
    return data_list

def gather(list):
    thresh = 20
    a = []
    b = []
    c = list[0]
    for d in list:
        if c + thresh > d:
            a.append(d)
        else:
            b.append(a)
            a = [d]
        c = d
    b.append(a)
    return b

def cluster(data_list):
    x_list = []
    y_list = []
    for data in data_list:
        x_list.append(data[1])
        y_list.append(data[2])

    x_list = list(set(x_list))
    x_list.sort()
    y_list = list(set(y_list))
    y_list.sort()

    x_list = gather(x_list)
    y_list = gather(y_list)

    return x_list, y_list

def position(data, list):
    for i, j in enumerate(list):
        if data in j:
            return i

def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))

        drop_score = 0.5
        dt_num = len(dt_boxes)

        ## 一种保存csv方法
        data_list = xy_info(dt_boxes, rec_res)
        if data_list:
            x_list, y_list = cluster(data_list)
            page = [[''] * len(x_list) for i in range(len(y_list))]
            for data in data_list:
                x = position(data[1], x_list)
                y = position(data[2], y_list)
                page[y][x] = data[0][0]

            file_path = os.path.splitext(image_file)[0] + '.xlsx'
            df = pd.DataFrame(page)
            df.to_excel(file_path, index=False)

        # ## 自制保存csv方法
        #
        # # 1.将识别结果和bncbox整合到一块
        # dt_boxes_np = np.array(dt_boxes)
        # dt_boxes_list = dt_boxes_np.tolist()
        # boxes = dt_boxes_list.copy()
        # for idx, box in enumerate(boxes):
        #     box.append(rec_res[idx])
        #
        # # 2.得到每行的row_dict，还有result_info，按行划分识别结果。
        # row = 1
        # row_dict = {}
        # cur_row_list = []
        # result_info = []
        # before_y_up = min(dt_boxes[0][0][1], dt_boxes[0][1][1])
        # before_y_down = max(dt_boxes[0][2][1], dt_boxes[0][3][1])
        # # 先将bounding box分行
        # for box in boxes:
        #     cur_y_up = min(box[0][1], box[1][1])
        #     cur_y_down = max(box[2][1], box[3][1])
        #     if cur_y_up - before_y_up <= 10 and cur_y_down - before_y_down <= 10:
        #         cur_row_list.append(box)
        #         before_y_up = cur_y_up
        #         before_y_down = cur_y_down
        #     else:
        #         ## 每行需进行内部排序
        #         cur_row_list = sorted(cur_row_list, key=lambda cur_row_list: cur_row_list[0][0])
        #         result_info.append([res[4][0] for res in cur_row_list])
        #         row_dict[str(row)] = cur_row_list
        #         row += 1
        #         cur_row_list = []
        #         cur_row_list.append(box)
        #         before_y_up = cur_y_up
        #         before_y_down = cur_y_down
        # cur_row_list = sorted(cur_row_list, key=lambda cur_row_list: cur_row_list[0][0])
        # result_info.append([res[4][0] for res in cur_row_list])
        # row_dict[str(row)] = cur_row_list
        #
        # row_num = len(row_dict)
        # col_num = max(len(row_dict[key]) for key in row_dict.keys())
        # # print("row_num: {}, col_num: {}".format(row_num, col_num))
        # len_list = [len(row_dict[key]) for key in row_dict.keys()]
        # max_col_index = len_list.index(max(len_list))
        # first_col_box = row_dict[str(max_col_index + 1)][0]
        # col_coord = []
        # for idx, box in enumerate(row_dict[str(max_col_index + 1)]):
        #     if idx == 0:
        #         each_col_corrd = min(box[0][0], box[3][0])
        #     elif idx == col_num - 1:
        #         each_col_corrd = min(box[0][0], box[3][0])
        #     else:
        #         each_col_corrd = min(box[0][0], box[3][0])
        #     col_coord.append(each_col_corrd)
        #
        # # 3.将识别结果按bndbox坐标顺序保存到pages
        # pages = [[''] * col_num for i in range(row_num)]
        # for key in row_dict:
        #     if len(row_dict[key]) == col_num:
        #         pages[int(key) - 1] = result_info[int(key) - 1]
        #     else:
        #         for idx, box in enumerate(row_dict[key]):
        #             temp_list = col_coord.copy()
        #             temp_list.append(box[0][0])
        #             temp_list.sort()
        #             cur_x_index = min(temp_list.index(box[0][0]), len(col_coord) - 1)
        #             if cur_x_index != 0:
        #                 pages[int(key) - 1][cur_x_index] = box[4][0]
        #
        # # 4.将pages识别结果保存为excel
        # file_path = os.path.splitext(image_file)[0] + '.xlsx'
        # df = pd.DataFrame(pages)
        # df.to_excel(file_path, index=False)



if __name__ == "__main__":
    print("doing!")
    main(args)