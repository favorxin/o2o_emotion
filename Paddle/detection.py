import os
import sys
import cv2
import copy
import numpy as np
import math
import time
from shapely.geometry import Polygon
import pyclipper
from copy import deepcopy
import logging
import argparse

import paddle.fluid as fluid
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

det_model_dir = 'D:/CV/code/PaddleOCR/PaddleOCR-develop/inference_large/det/'

# def parse_args():
#     def str2bool(v):
#         return v.lower() in ("true", "t", "1")
#
#     parser = argparse.ArgumentParser()
#     # params for prediction engine
#     parser.add_argument("--use_gpu", type=str2bool, default=False)
#     parser.add_argument("--ir_optim", type=str2bool, default=True)
#     parser.add_argument("--use_tensorrt", type=str2bool, default=False)
#     parser.add_argument("--gpu_mem", type=int, default=8000)
#
#     # params for text detector
#     parser.add_argument("--image_dir", type=str)
#     parser.add_argument("--det_algorithm", type=str, default='DB')
#     parser.add_argument("--det_model_dir", type=str, default=det_model_dir)
#     parser.add_argument("--det_max_side_len", type=float, default=960)
#
#     # DB parmas
#     parser.add_argument("--det_db_thresh", type=float, default=0.3)
#     parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
#     parser.add_argument("--det_db_unclip_ratio", type=float, default=2.0)
#
#     # EAST parmas
#     parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
#     parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
#     parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)
#
#     # SAST parmas
#     parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
#     parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
#     parser.add_argument("--det_sast_polygon", type=bool, default=False)
#
#     # params for text recognizer
#     parser.add_argument("--rec_algorithm", type=str, default='CRNN')
#     parser.add_argument("--rec_model_dir", type=str)
#     parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
#     parser.add_argument("--rec_char_type", type=str, default='ch')
#     parser.add_argument("--rec_batch_num", type=int, default=30)
#     parser.add_argument("--max_text_length", type=int, default=25)
#     parser.add_argument("--rec_char_dict_path", type=str)
#     parser.add_argument("--use_space_char", type=bool, default=True)
#     parser.add_argument("--enable_mkldnn", type=bool, default=False)
#     parser.add_argument("--use_zero_copy_run", type=bool, default=False)
#     return parser.parse_args()

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

class DBProcessTest(object):
    """
    DB pre-process for Test mode
    """

    def __init__(self, params):
        super(DBProcessTest, self).__init__()
        self.resize_type = 0
        if 'test_image_shape' in params:
            self.image_shape = params['test_image_shape']
            # print(self.image_shape)
            self.resize_type = 1
        if 'max_side_len' in params:
            self.max_side_len = params['max_side_len']
        else:
            self.max_side_len = 2400

    def resize_image_type0(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = self.max_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image_type1(self, im):
        """
        resize image to a size self.image_shape
        :param im: input image
        :return: normalized image and resize ratio
        """
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = im.shape[:2]  # (h, w, c)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return im, (ratio_h, ratio_w)

    def normalize(self, im):
        """
        Normalize image
        :param im: input image
        :return: Normalized image
        """
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def __call__(self, im):
        if self.resize_type == 0:
            im, (ratio_h, ratio_w) = self.resize_image_type0(im)
        else:
            im, (ratio_h, ratio_w) = self.resize_image_type1(im)
        im = self.normalize(im)
        im = im[np.newaxis, :]
        return [im, (ratio_h, ratio_w)]

class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, params):
        self.thresh = params['thresh']
        self.box_thresh = params['box_thresh']
        self.max_candidates = params['max_candidates']
        self.unclip_ratio = params['unclip_ratio']
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, mask):
        """
        Get boxes from the binarized image predicted by DB.
        :param pred: the binarized image predicted by DB.
        :param mask: new 'pred' after threshold filtering.
        :return: (boxes, the score of each boxes)
        """
        dest_height, dest_width = pred.shape[-2:]
        bitmap = deepcopy(mask)
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box):
        """
        Shrink or expand the boxaccording to 'unclip_ratio'
        :param box: The predicted box.
        :return: uncliped box
        """
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        """
        Get boxes from the contour or box.
        :param contour: The predicted contour.
        :return: The predicted box.
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        Calculate the score of box.
        :param bitmap: The binarized image predicted by DB.
        :param _box: The predicted box
        :return: score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, ratio_list):
        pred = outs_dict['maps']

        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):

            mask = cv2.dilate(
                np.array(segmentation[batch_index]).astype(np.uint8),
                self.dilation_kernel)
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(pred[batch_index],
                                                           mask)

            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
        return boxes_batch


class Detecting(object):
    def __init__(self, args):
        max_side_len = args.det_max_side_len
        self.det_algorithm = args.det_algorithm
        preprocess_params = {'max_side_len': max_side_len}
        postprocess_params = {}
        self.preprocess_op = DBProcessTest(preprocess_params)
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        self.postprocess_op = DBPostProcess(postprocess_params)
        self.use_zero_copy_run = args.use_zero_copy_run
        self.predictor, self.input_tensor, self.output_tensors =\
            create_predictor(args, mode="det")

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        im = im.copy()
        starttime = time.time()
        im = fluid.core.PaddleTensor(im)
        self.predictor.run([im])
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        outs_dict = {}

        outs_dict['maps'] = outputs[0]
        dt_boxes_list = self.postprocess_op(outs_dict, [ratio_list])
        dt_boxes = dt_boxes_list[0]
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime

        return dt_boxes, elapse




