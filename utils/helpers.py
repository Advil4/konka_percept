"""
    helper functions for 

"""

import cv2
import numpy as np


def mask_T_contour(mask):
    """
        takes in an uint 8 mask and returns a contours array
    """
    mask = mask.astype(np.uint8)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contour


def letterbox(im, new_shape=(416, 416), color=(114, 114, 114)):
    """
    helper function for detection model 

    调整图像尺寸，保持纵横比，并填充边界。

    参数:
    - im: 输入图像(ndarray)
    - new_shape: 新的尺寸，默认为(416, 416)
    - color: 填充颜色（默认为(114, 114, 114))

    输出:
    - im: 调整大小后的图像(ndarray)
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def filter_box(outputs, class_names, score_threshold, nms_threshold):
    """
    helper function which filters out useless detection boxes and performs Non-Maximum Suppression (NMS).
    过滤无用的检测框，并执行NMS（非极大值抑制）。

    参数:
    - outputs: 模型输出的预测框(ndarray)
    - class_names: 类别名称(list)
    - score_threshold: 置信度阈值
    - nms_threshold: NMS阈值

    输出:
    - boxes: 经过筛选和NMS后的检测框(ndarray)
    """
    outputs = np.squeeze(outputs)

    boxes = []
    scores = []
    class_ids = []
    classes_scores = outputs[4:(4 + len(class_names)), ...]

    for i in range(outputs.shape[1]):
        class_id = np.argmax(classes_scores[..., i])
        score = classes_scores[class_id][i]
        if score > score_threshold:
            boxes.append(np.concatenate([outputs[:4, i], np.array([score, class_id])]))
            scores.append(score)
            class_ids.append(class_id)
    if boxes != []:
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, score_threshold, nms_threshold)
        boxes = boxes[indices]
    else:
        boxes = np.array(boxes)
    return boxes


def nms(boxes, scores, score_threshold, nms_threshold):
    """
    helper function which performs Non-Maximum Suppression (NMS) used in the above function.

    执行非极大值抑制（NMS）。

    参数:
    - boxes: 输入框（ndarray）
    - scores: 框的置信度（ndarray）
    - score_threshold: 置信度阈值
    - nms_threshold: NMS阈值

    输出:
    - keep: 保留的框的索引（list）
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= nms_threshold)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    """
    helper function which converts bounding boxes from xywh to xyxy format.
    将边界框从中心宽高格式（xywh）转换为坐标格式（xyxy）。

    参数:
    - x: 输入框（ndarray）

    输出:
    - y: 转换后的框（ndarray）
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_boxes(boxes, input_shape, output_shape):
    """
    helper function which scales bounding boxes from xywh to xyxy format.
    对边界框进行缩放，以适应新图像尺寸。
        
    参数:
    - boxes: 输入框（ndarray）
    - input_shape: 输入图像的原始尺寸
    - output_shape: 目标图像的尺寸

    输出:
    - boxes: 缩放后的框（ndarray）
    """
    # Rescale boxes (xyxy) from self.input_shape to shape
    gain = min(input_shape[0] / output_shape[0], input_shape[1] / output_shape[1])  # gain  = old / new
    pad = (input_shape[1] - output_shape[1] * gain) / 2, (input_shape[0] - output_shape[0] * gain) / 2  # wh padding
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, output_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, output_shape[0])  # y1, y2
    return boxes


def get_color(track_id):
    """
    helper function which generates a unique color for each track ID
    生成每个跟踪ID的唯一颜色。

    参数:
    - track_id: 跟踪ID（int）

    输出:
    - color: 生成的RGB颜色元组（tuple）
    """
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())


def load_vocab(path="en_vocabulary.txt"):
    with open(path, 'r') as f:
        vocab = [line.strip() for line in f.readlines()]
    return vocab
