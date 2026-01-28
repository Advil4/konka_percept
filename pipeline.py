import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from boxmot import BotSort

from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.utils.torch_utils import select_device
from utils.dual_branch_predictor import DualBranchPredictor
from utils.dynamic_predictor import DynamicDetectionPredictor, DynamicSegmentationPredictor
from utils.helpers import mask_T_contour
from utils.logger import SingletonLogger
from utils.tensorrt_backend import TensorRTBackend

# 配置日志系统
logger = SingletonLogger(
    name='pipeline',
    log_file='logs/pipeline.log',
    level='INFO',
    console=True
)


@dataclass
class PipelineData:
    """统一的数据结构用于在各阶段之间传递"""
    frame: np.ndarray  # 原始图像
    im: torch.Tensor  # 预处理后的 Tensor
    timestamp: float

    infer_results: Optional[Any] = None
    post_results: Optional[Any] = None
    detections: Optional[np.ndarray] = None
    all_tracked_objects: Optional[Any] = None
    tracked_objects: Optional[Any] = None
    masks_coords: Optional[Any] = None
    mask_sizes: Optional[Any] = None
    feature_maps: Optional[Any] = None

    pre_time: float = 0.0
    infer_time: float = 0.0
    post_time: float = 0.0
    track_time: float = 0.0
    result_time: float = 0.0


class PipelineStage(threading.Thread):
    """所有阶段的基础类，使用队列驱动模式"""

    def __init__(self, input_queue, output_queue=None, max_retries=3, timeout=0.1):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_flag = False
        self.max_retries = max_retries
        self.timeout = timeout

    def stop(self):
        self.stop_flag = True

    def is_stopped(self):
        return self.stop_flag


class InferenceStage(PipelineStage):
    def __init__(self, pipeline, input_queue, output_queue):
        super().__init__(input_queue, output_queue)
        self.pipeline = pipeline

    def run(self):
        while not self.is_stopped():
            try:
                # 阻塞式获取数据
                data = self.input_queue.get(timeout=self.timeout)

                # 深拷贝输入避免内存污染
                safe_input = data.im.clone().to(self.pipeline.device)

                start = time.time()
                preds = self.pipeline.tensorrt_model(safe_input)

                # 更新推理时间
                data.infer_time = (time.time() - start) * 1e3
                data.infer_results = preds

                # 传递到下一阶段
                if self.output_queue:
                    self.output_queue.put(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"推理阶段异常: {e}")
                continue


class PostProcessStage(PipelineStage):
    def __init__(self, pipeline, input_queue, output_queue):
        super().__init__(input_queue, output_queue)
        self.pipeline = pipeline

    def run(self):
        while not self.is_stopped():
            try:
                data = self.input_queue.get(timeout=self.timeout)

                start = time.time()
                predictor = self.pipeline.predictor
                predictor.args.retina_masks = True
                predictor.args.agnostic_nms = True
                predictor.args.iou = self.pipeline.cfg['model_cfg']['iou_threshold']
                predictor.args.conf = self.pipeline.cfg['model_cfg']['conf_threshold']

                # 处理双分支模型的输出
                infer_results = data.infer_results

                # 检查是否是7个输出的双分支模型（包含特征图）
                if isinstance(infer_results, (list, tuple)) and len(infer_results) > 4:
                    general_detect, general_segment, gripper_detect, gripper_segment, feature_list = infer_results
                    feat_20, feat_40, feat_80 = feature_list

                    # 保存特征图信息
                    feature_maps = [feat_80, feat_40, feat_20]
                    predictor._feats = feature_maps

                    # 对两个分支的结果分别进行后处理
                    import itertools
                    items_list = list(predictor.model.names.items())
                    n_items = len(items_list)
                    all_names_dict = dict(items_list)

                    # 保存原始names
                    original_names = predictor.model.names

                    try:
                        # 设置通用分支的类别名称
                        predictor.model.names = dict(itertools.islice(all_names_dict.items(), n_items - 1))
                        general_results = predictor.postprocess([general_detect, general_segment], data.im,
                                                                [data.frame])
                    except Exception as e:
                        logger.warning(f"通用分支后处理失败: {e}")
                        general_results = None

                    try:
                        # 设置夹爪分支的类别名称
                        predictor.model.names = dict(itertools.islice(all_names_dict.items(), n_items - 1, n_items))
                        gripper_results = predictor.postprocess([gripper_detect, gripper_segment], data.im,
                                                                [data.frame])
                    except Exception as e:
                        logger.warning(f"夹爪分支后处理失败: {e}")
                        gripper_results = None

                    # 恢复原始names
                    predictor.model.names = original_names

                    # 修正夹爪类别的类别索引
                    if gripper_results and len(gripper_results) > 0:
                        gripper_result = gripper_results[0]
                        if hasattr(gripper_result, 'cls') and gripper_result.cls is not None:
                            # 将夹爪类别的索引从0改为总类别数-1
                            gripper_result.cls.fill_(n_items - 1)

                        # 如果有boxes对象，也需要更新类别
                        if hasattr(gripper_result, 'boxes') and gripper_result.boxes is not None:
                            if hasattr(gripper_result.boxes, 'cls') and gripper_result.boxes.cls is not None:
                                gripper_result.boxes.cls.fill_(n_items - 1)

                    # 返回两个分支的结果
                    results = [general_results, gripper_results]
                    data.post_time = (time.time() - start) * 1e3

                    data.post_results = [2, results]
                else:
                    # 单分支模型输出或非双分支情况
                    results = predictor.postprocess(infer_results, data.im, [data.frame])
                    data.post_time = (time.time() - start) * 1e3
                    data.post_results = [1, results]

                # 传递到下一阶段
                if self.output_queue:
                    self.output_queue.put(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"后处理阶段异常: {e}")
                import traceback
                traceback.print_exc()
                continue


class TrackingStage(PipelineStage):
    def __init__(self, pipeline, input_queue, output_queue):
        super().__init__(input_queue, output_queue)
        self.pipeline = pipeline

    def run(self):
        while not self.is_stopped():
            try:
                data = self.input_queue.get(timeout=self.timeout)

                if not data.post_results or len(data.post_results) == 0:
                    continue

                start = time.time()

                # 检查是否是双分支模型的结果
                if data.post_results[0] >= 2:
                    # 双分支模型，分别处理跟踪
                    general_tracked_objects, gripper_tracked_objects = self._track_dual_branch(data)
                    # 合并跟踪结果
                    if len(general_tracked_objects) > 0 and len(gripper_tracked_objects) > 0:
                        all_tracked_objects = general_tracked_objects + gripper_tracked_objects
                    elif len(general_tracked_objects) > 0:
                        # 添加一个占位符对象以保持数组结构
                        placeholder_obj = [0, 0, 0, 0, -1, 0, 0, 0, -1, []]  # 占位符对象
                        all_tracked_objects = general_tracked_objects + [placeholder_obj]
                    elif len(gripper_tracked_objects) > 0:
                        # 添加一个占位符对象以保持数组结构
                        placeholder_obj = [0, 0, 0, 0, -1, 0, 0, 0, -1, []]  # 占位符对象
                        all_tracked_objects = [placeholder_obj] + gripper_tracked_objects
                    else:
                        all_tracked_objects = []
                else:
                    # 单分支模型，使用原有逻辑
                    boxes = data.post_results[1][0].boxes.cpu().numpy()

                    xyxy = boxes.xyxy.copy()

                    # 确保 xyxy 的每个坐标都不小于 0，并且不超过图像的宽度和高度
                    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, self.pipeline.orig_w)  # x1
                    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, self.pipeline.orig_h)  # y1
                    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, self.pipeline.orig_w)  # x2
                    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, self.pipeline.orig_h)  # y2

                    # 拼接检测结果
                    detections = np.hstack([
                        xyxy,
                        boxes.conf.reshape(-1, 1),
                        boxes.cls.reshape(-1, 1),
                    ])

                    data.detections = detections
                    tracked_objects_old = self.pipeline.tracker.update(detections, data.frame)
                    updated_tracked_objects = []
                    for obj in tracked_objects_old:
                        detection_idx = int(obj[7])
                        new_bbox = detections[detection_idx][:4]
                        updated_obj = obj.copy()
                        updated_obj[:4] = new_bbox
                        updated_tracked_objects.append(updated_obj)
                    all_tracked_objects = np.array(updated_tracked_objects)

                data.all_tracked_objects = all_tracked_objects
                data.track_time = (time.time() - start) * 1e3

                # 传递到下一阶段
                if self.output_queue:
                    self.output_queue.put(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"跟踪阶段异常: {e}")
                continue

    def _track_dual_branch(self, data):
        """
        对双分支模型的结果分别进行跟踪
        """
        try:
            # 获取后处理结果
            general_post_results = data.post_results[1][0]

            # 安全地获取通用分支的boxes
            general_boxes_result = general_post_results[0]
            if hasattr(general_boxes_result, 'boxes'):
                general_boxes = general_boxes_result.boxes
                # 安全地转换为numpy数组
                if hasattr(general_boxes, 'cpu'):
                    general_boxes_np = general_boxes.cpu().numpy()
                elif hasattr(general_boxes, 'numpy'):
                    general_boxes_np = general_boxes.numpy()
                else:
                    general_boxes_np = np.array(general_boxes)
            else:
                general_boxes_np = np.array(general_boxes_result)

            # 安全地获取特征向量（如果存在）
            if hasattr(general_boxes_result, 'feats') and general_boxes_result.feats is not None:
                general_feats = general_boxes_result.feats[0]
                general_feats_idx = general_boxes_result.feats[1]
                # 转换为numpy数组
                if hasattr(general_feats, 'cpu'):
                    general_feats = general_feats.cpu().numpy()
                else:
                    general_feats = np.array(general_feats)

                if hasattr(general_feats_idx, 'cpu'):
                    general_feats_idx = general_feats_idx.cpu().numpy()
                else:
                    general_feats_idx = np.array(general_feats_idx)
            else:
                general_feats = None
                general_feats_idx = None

            general_detections = self._create_detections_safe(general_boxes_np)
            general_tracked_objects_old = self.pipeline.general_tracker.update(general_detections, data.frame)
            updated_general_tracked_objects = []
            for obj in general_tracked_objects_old:
                detection_idx = int(obj[7])
                new_bbox = general_detections[detection_idx][:4]
                # 安全处理特征
                new_feats = general_feats[detection_idx].tolist() if general_feats is not None else []
                new_feats_idx = int(general_feats_idx[detection_idx]) if general_feats_idx is not None else -1
                updated_obj = obj.copy().tolist()
                updated_obj[:4] = new_bbox
                updated_obj.append(new_feats_idx)
                updated_obj.append(new_feats)
                updated_general_tracked_objects.append(updated_obj)
            general_tracked_objects = updated_general_tracked_objects

            # 处理夹爪分支
            gripper_post_results = data.post_results[1][1]
            # 添加安全检查
            if gripper_post_results and len(gripper_post_results) > 0 and gripper_post_results[0] is not None:
                gripper_boxes_result = gripper_post_results[0]
                if hasattr(gripper_boxes_result, 'boxes'):
                    gripper_boxes = gripper_boxes_result.boxes
                    # 安全地转换为numpy数组
                    if hasattr(gripper_boxes, 'cpu'):
                        gripper_boxes_np = gripper_boxes.cpu().numpy()
                    elif hasattr(gripper_boxes, 'numpy'):
                        gripper_boxes_np = gripper_boxes.numpy()
                    else:
                        gripper_boxes_np = np.array(gripper_boxes)
                else:
                    gripper_boxes_np = np.array(gripper_boxes_result)

                # 安全地获取特征向量（如果存在）
                if hasattr(gripper_boxes_result, 'feats') and gripper_boxes_result.feats is not None:
                    gripper_feats = gripper_boxes_result.feats[0]
                    gripper_feats_idx = gripper_boxes_result.feats[1]
                    # 转换为numpy数组
                    if hasattr(gripper_feats, 'cpu'):
                        gripper_feats = gripper_feats.cpu().numpy()
                    else:
                        gripper_feats = np.array(gripper_feats)

                    if hasattr(gripper_feats_idx, 'cpu'):
                        gripper_feats_idx = gripper_feats_idx.cpu().numpy()
                    else:
                        gripper_feats_idx = np.array(gripper_feats_idx)
                else:
                    gripper_feats = None
                    gripper_feats_idx = None

                gripper_detections = self._create_detections_safe(gripper_boxes_np)
                gripper_tracked_objects_old = self.pipeline.gripper_tracker.update(gripper_detections, data.frame)
                updated_gripper_tracked_objects = []
                for obj in gripper_tracked_objects_old:
                    detection_idx = int(obj[7])
                    new_bbox = gripper_detections[detection_idx][:4]
                    # 安全处理特征
                    new_feats = gripper_feats[detection_idx].tolist() if gripper_feats is not None else []
                    new_feats_idx = int(gripper_feats_idx[detection_idx]) if gripper_feats_idx is not None else -1
                    updated_obj = obj.copy().tolist()
                    updated_obj[:4] = new_bbox
                    updated_obj.append(new_feats_idx)
                    updated_obj.append(new_feats)
                    updated_gripper_tracked_objects.append(updated_obj)
                gripper_tracked_objects = updated_gripper_tracked_objects
            else:
                gripper_tracked_objects = []
                gripper_detections = np.array([])

            if general_detections.size > 0 and gripper_detections.size > 0:
                data.detections = np.vstack([general_detections, gripper_detections]).astype(np.float32)
            elif general_detections.size > 0:
                # 创建与general_detections相同形状的占位符数组
                placeholder = np.zeros((1, general_detections.shape[1]), dtype=np.float32)
                data.detections = np.vstack([general_detections, placeholder]).astype(np.float32)
            elif gripper_detections.size > 0:
                # 创建与gripper_detections相同形状的占位符数组
                placeholder = np.zeros((1, gripper_detections.shape[1]), dtype=np.float32)
                data.detections = np.vstack([placeholder, gripper_detections]).astype(np.float32)
            else:
                data.detections = np.array([])

            return general_tracked_objects, gripper_tracked_objects

        except Exception as e:
            logger.error(f"双分支跟踪处理异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def _create_detections_safe(self, boxes_data):
        """
        安全地从boxes数据创建detections数组
        """
        try:
            # 确保boxes_data是numpy数组
            if hasattr(boxes_data, 'cpu'):
                boxes_np = boxes_data.cpu().numpy()
            elif hasattr(boxes_data, 'numpy'):
                boxes_np = boxes_data.numpy()
            else:
                boxes_np = np.array(boxes_data)

            # 获取xyxy坐标
            if hasattr(boxes_data, 'xyxy'):
                if hasattr(boxes_data.xyxy, 'cpu'):
                    xyxy = boxes_data.xyxy.cpu().numpy().copy()
                else:
                    xyxy = boxes_data.xyxy.copy()
            else:
                # 假设boxes_np已经是xyxy格式
                xyxy = boxes_np.copy()
                if xyxy.ndim == 2 and xyxy.shape[1] >= 4:
                    xyxy = xyxy[:, :4]

            # 确保 xyxy 的每个坐标都不小于 0，并且不超过图像的宽度和高度
            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, self.pipeline.orig_w)  # x1
            xyxy[:, 1] = np.clip(xyxy[:, 1], 0, self.pipeline.orig_h)  # y1
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, self.pipeline.orig_w)  # x2
            xyxy[:, 3] = np.clip(xyxy[:, 3], 0, self.pipeline.orig_h)  # y2

            # 获取置信度和类别
            if hasattr(boxes_data, 'conf') and hasattr(boxes_data, 'cls'):
                if hasattr(boxes_data.conf, 'cpu'):
                    conf_data = boxes_data.conf.cpu().numpy().reshape(-1, 1)
                else:
                    conf_data = boxes_data.conf.reshape(-1, 1)

                if hasattr(boxes_data.cls, 'cpu'):
                    cls_data = boxes_data.cls.cpu().numpy().reshape(-1, 1)
                else:
                    cls_data = boxes_data.cls.reshape(-1, 1)
            else:
                # 如果没有conf和cls属性，使用默认值或从boxes_np中提取
                if boxes_np.ndim == 2 and boxes_np.shape[1] >= 6:
                    conf_data = boxes_np[:, 4].reshape(-1, 1)
                    cls_data = boxes_np[:, 5].reshape(-1, 1)
                else:
                    # 默认值
                    conf_data = np.ones((xyxy.shape[0], 1))
                    cls_data = np.zeros((xyxy.shape[0], 1))

            # 拼接检测结果
            detections = np.hstack([
                xyxy,
                conf_data,
                cls_data
            ])

            return detections.astype(np.float32)

        except Exception as e:
            logger.error(f"创建detections时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([])

    def _create_detections(self, boxes_data):
        """
        从boxes对象创建detections数组（兼容旧版本）
        """
        try:
            # 首先尝试使用安全方法
            return self._create_detections_safe(boxes_data)
        except Exception as e:
            logger.warning(f"使用安全方法创建detections失败，回退到旧方法: {e}")
            # 回退到原始实现
            if hasattr(boxes_data, 'xyxy'):
                xyxy = boxes_data.xyxy.copy()
            else:
                # 如果没有xyxy属性，假设boxes_data本身就是xyxy格式
                xyxy = boxes_data.copy()
                if xyxy.ndim == 2 and xyxy.shape[1] >= 4:
                    xyxy = xyxy[:, :4]

            # 确保 xyxy 的每个坐标都不小于 0，并且不超过图像的宽度和高度
            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, self.pipeline.orig_w)  # x1
            xyxy[:, 1] = np.clip(xyxy[:, 1], 0, self.pipeline.orig_h)  # y1
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, self.pipeline.orig_w)  # x2
            xyxy[:, 3] = np.clip(xyxy[:, 3], 0, self.pipeline.orig_h)  # y2

            # 获取置信度和类别（回退方法）
            try:
                conf_data = boxes_data.conf.reshape(-1, 1)
                cls_data = boxes_data.cls.reshape(-1, 1)
            except AttributeError:
                # 如果无法获取conf和cls，使用默认值
                conf_data = np.ones((xyxy.shape[0], 1))
                cls_data = np.zeros((xyxy.shape[0], 1))

            # 拼接检测结果
            detections = np.hstack([
                xyxy,
                conf_data,
                cls_data
            ])

            return detections.astype(np.float32)


class ResultProcessingStage(PipelineStage):
    """对跟踪后的数据进行最终处理（如稳定目标过滤、mask 提取等）"""

    def __init__(self, pipeline, input_queue, output_queue):
        super().__init__(input_queue, output_queue)
        self.pipeline = pipeline
        vocab_path = "./en_vocabulary.txt"
        with open(vocab_path, 'r') as f:
            self.all_class_name = [line.strip() for line in f.readlines()]
        move_vocab_path = "./en_move_vocabulary.txt"
        with open(move_vocab_path, 'r') as f:
            self.move_class_name = [line.strip() for line in f.readlines()]

    def run(self):
        while not self.is_stopped():
            try:
                data = self.input_queue.get(timeout=self.timeout)

                # 提取基础信息
                start = time.time()
                all_detections = data.detections
                all_masks = self._get_masks_from_data(data)

                if data.post_results[0] >= 2:
                    # 筛选稳定目标
                    general_stable_detections, general_stable_indices = self._filter_stable_objects(
                        data.all_tracked_objects[:-1],
                        int(data.timestamp))
                    gripper_stable_detections, gripper_stable_indices = self._filter_stable_objects(
                        data.all_tracked_objects[-1:],
                        int(data.timestamp))

                    general_tracked_objects = self._build_tracked_objects(general_stable_detections,
                                                                          int(data.timestamp))
                    gripper_tracked_objects = self._build_tracked_objects(gripper_stable_detections,
                                                                          int(data.timestamp))
                    if general_tracked_objects and gripper_tracked_objects:
                        data.tracked_objects = general_tracked_objects + gripper_tracked_objects
                    elif general_tracked_objects:
                        data.tracked_objects = general_tracked_objects
                    elif gripper_tracked_objects:
                        data.tracked_objects = gripper_tracked_objects
                    else:
                        data.tracked_objects = []

                    if self.pipeline.task == 'segment':
                        general_masks_coords, general_mask_sizes, general_masks = self._process_masks(data,
                                                                                                      all_masks[:-1, :],
                                                                                                      all_detections[
                                                                                                      :-1, :],
                                                                                                      general_stable_indices)
                        if gripper_tracked_objects:
                            gripper_masks_coords, gripper_mask_sizes, gripper_masks = self._process_masks(data,
                                                                                                          all_masks[-1:,
                                                                                                          :],
                                                                                                          all_detections[
                                                                                                          -1:, :],
                                                                                                          gripper_stable_indices)
                            general_masks_coords, general_mask_sizes = self._filter_object(data, general_masks,
                                                                                           gripper_masks,
                                                                                           general_masks_coords,
                                                                                           general_mask_sizes)
                            data.masks_coords = general_masks_coords + gripper_masks_coords
                            data.mask_sizes = general_mask_sizes + gripper_mask_sizes
                        else:
                            data.masks_coords = general_masks_coords
                            data.mask_sizes = general_mask_sizes

                else:
                    stable_detections, stable_indices = self._filter_stable_objects(
                        data.all_tracked_objects[-1:, :],
                        int(data.timestamp))

                    # 构建 tracked_objects
                    data.tracked_objects = self._build_tracked_objects(stable_detections, int(data.timestamp))

                    # 处理 mask 数据（仅在分割任务中）
                    if self.pipeline.task == 'segment':
                        masks_coords, mask_sizes = self._process_masks(data, all_masks, all_detections, stable_indices)
                        data.masks_coords = masks_coords
                        data.mask_sizes = mask_sizes

                data.result_time = (time.time() - start) * 1e3

                # 更新帧计数并输出
                self.pipeline.frame_count += 1
                if self.output_queue:
                    self.output_queue.put(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"结果处理阶段异常: {e}")
                continue

    def _add_features_to_tracked_objects(self, tracked_objects, detection_result, feature_maps):
        """
        为跟踪对象添加对应的特征向量

        Args:
            tracked_objects: 跟踪对象列表
            detection_result: 检测结果，形状为[xyxy+类别分数+32掩码系数, 8400]
            feature_maps: 特征图张量，形状为[1, 512, 总特征点数]

        Returns:
            list: 添加了特征向量的跟踪对象列表
        """
        if detection_result is None or feature_maps is None:
            return [list(obj) + [None] for obj in tracked_objects]

        detection_result = detection_result.transpose(-1, -2)
        detection_result[..., :4] = xywh2xyxy(detection_result[..., :4])
        detection_result = detection_result.squeeze(0)
        detection_result[:, :4] = scale_boxes((640, 640), detection_result[:, :4], (640, 480, 3))

        # 从detection_result中提取边界框坐标
        xyxy_coords = detection_result[:, :4]
        xyxy_coords_array = xyxy_coords.cpu().numpy()
        xyxy_coords_array[:, 0] = np.clip(xyxy_coords_array[:, 0], 0, self.pipeline.orig_w)  # x1
        xyxy_coords_array[:, 1] = np.clip(xyxy_coords_array[:, 1], 0, self.pipeline.orig_h)  # y1
        xyxy_coords_array[:, 2] = np.clip(xyxy_coords_array[:, 2], 0, self.pipeline.orig_w)  # x2
        xyxy_coords_array[:, 3] = np.clip(xyxy_coords_array[:, 3], 0, self.pipeline.orig_h)  # y2

        detection_boxes = []
        for i in range(xyxy_coords.shape[0]):
            x1, y1, x2, y2 = xyxy_coords[i, :]
            detection_boxes.append((float(x1), float(y1), float(x2), float(y2)))

        # 为每个跟踪对象添加特征向量
        enhanced_tracked_objects = []
        for tracked_obj in tracked_objects:
            # 获取跟踪对象的边界框
            tracked_box = (float(tracked_obj[0]), float(tracked_obj[1]),
                           float(tracked_obj[2]), float(tracked_obj[3]))

            # 查找精确匹配的检测框
            matched_index = None
            for i, detection_box in enumerate(detection_boxes):
                # 直接比较四个值是否相等（考虑浮点数精度）
                if (abs(detection_box[0] - tracked_box[0]) < 10 and
                        abs(detection_box[1] - tracked_box[1]) < 10 and
                        abs(detection_box[2] - tracked_box[2]) < 10 and
                        abs(detection_box[3] - tracked_box[3]) < 10):
                    matched_index = i
                    break

            if matched_index is not None:
                # 从feature_maps中提取对应的特征向量
                # feature_maps的形状应该是[1, 512, N]，其中N是特征点总数
                try:
                    # 提取第matched_index列的所有行（512个特征）
                    feature_vector = feature_maps[0, :, matched_index].cpu().numpy()  # 形状: [512]
                    # 将特征向量添加到跟踪对象中
                    enhanced_obj = list(tracked_obj) + [feature_vector]
                    enhanced_tracked_objects.append(enhanced_obj)
                    logger.info(f"添加特征向量成功")
                except Exception as e:
                    logger.warning(f"提取特征向量时出错: {e}")
                    enhanced_tracked_objects.append(list(tracked_obj) + [None])
            else:
                # 如果没有匹配的检测框，添加空特征向量
                enhanced_tracked_objects.append(list(tracked_obj) + [None])
                logger.warning(f"没有匹配的检测框，添加空特征向量")

        return enhanced_tracked_objects

    def _build_tracked_objects(self, stable_detections, timestamp):
        """根据稳定检测结果构建被跟踪对象数组。"""
        if not stable_detections:
            return []

        tracked_objects = []
        for item in stable_detections:
            # 跳过占位符对象
            if isinstance(item, (list, np.ndarray)) and len(item) > 4:
                track_id = int(item[4]) if isinstance(item[4], (int, float)) else -1
                if track_id < 0:  # 跳过占位符
                    continue

                x1, y1, x2, y2, tid, conf, cls_idx, idx, feats_idx, feats = item
                # 优化坐标限制逻辑
                x1_clamped = max(0, min(x1, self.pipeline.orig_w))
                y1_clamped = max(0, min(y1, self.pipeline.orig_h))
                x2_clamped = max(0, min(x2, self.pipeline.orig_w))
                y2_clamped = max(0, min(y2, self.pipeline.orig_h))

                track_warmup_info = self.pipeline.track_warmup.get(tid, f"{cls_idx}_{timestamp}")
                track_id_str = f"{track_warmup_info.split('_')[-1]}_{int(tid)}"

                tracked_objects.append((
                    x1_clamped, y1_clamped, x2_clamped, y2_clamped,
                    track_id_str, conf, cls_idx, idx, feats_idx, feats
                ))

        return tracked_objects

    def _filter_stable_objects(self, all_tracked_objects, timestamp):
        """过滤出稳定跟踪的对象。"""
        # 添加输入验证和占位符过滤
        if not all_tracked_objects:
            return [], []

        # 过滤掉占位符对象（跟踪ID为负数的对象）
        filtered_objects = []
        for obj in all_tracked_objects:
            # 检查是否为占位符对象（跟踪ID为-1）
            if isinstance(obj, (list, np.ndarray)) and len(obj) > 4:
                track_id = int(obj[4]) if isinstance(obj[4], (int, float)) else -1
                if track_id >= 0:  # 只处理有效的跟踪对象
                    filtered_objects.append(obj)
            else:
                # 对于非标准格式的对象，尝试处理
                try:
                    track_id = int(obj[4])
                    if track_id >= 0:
                        filtered_objects.append(obj)
                except (IndexError, ValueError, TypeError):
                    continue  # 跳过无效对象

        if not filtered_objects:
            return [], []

        stable_detections = []
        stable_indices = []

        for obj in filtered_objects:
            track_id = int(obj[4])
            cls_idx = int(obj[6])

            if track_id not in self.pipeline.track_warmup:
                self.pipeline.track_warmup[track_id] = f"{cls_idx}_{timestamp}"
                self.pipeline.track_appearances[track_id] = 1
            else:
                self.pipeline.track_appearances[track_id] += 1

            if self.pipeline.track_appearances[track_id] >= self.pipeline.track_filter:
                self.pipeline.stable_tracks.add(track_id)
                obj[6] = int(self.pipeline.track_warmup[track_id].split("_")[0])
                stable_detections.append(obj)
                stable_indices.append(int(obj[7]))

        return stable_detections, stable_indices

    def _get_masks_from_data(self, data):
        """从数据对象中获取掩膜信息。

        此方法检查数据对象的处理结果中是否包含掩膜信息，并返回这些信息。
        如果数据对象的处理结果中不包含掩膜信息或数据为空，则返回None。

        参数:
            data: 包含处理结果的数据对象。

        返回:
            如果存在掩膜信息，则返回掩码数据。
            如果不存在掩膜信息，则返回None。
        """
        try:
            if data.post_results[0] >= 2:
                # 检查数据是否存在，避免访问None对象的属性
                if (data.post_results[1] and len(data.post_results[1]) > 1 and
                        data.post_results[1][0] and len(data.post_results[1][0]) > 0 and data.post_results[1][0][
                            0] and hasattr(data.post_results[1][0][0], 'masks') and
                        data.post_results[1][1] and len(data.post_results[1][1]) > 0 and data.post_results[1][1][
                            0] and hasattr(data.post_results[1][1][0], 'masks')):
                    return torch.cat([data.post_results[1][0][0].masks.data, data.post_results[1][1][0].masks.data],
                                     dim=0)
                elif (data.post_results[1] and len(data.post_results[1]) > 0 and
                      data.post_results[1][0] and len(data.post_results[1][0]) > 0 and data.post_results[1][0][
                          0] and hasattr(data.post_results[1][0][0], 'masks')):
                    # 只有通用分支有mask数据，创建一个与第一个mask相同形状的零张量作为占位符
                    general_mask_data = data.post_results[1][0][0].masks.data
                    # 创建占位符mask，形状与第一个mask相同但只有一行
                    placeholder_mask = torch.zeros((1, general_mask_data.shape[1], general_mask_data.shape[2]),
                                                   dtype=general_mask_data.dtype,
                                                   device=general_mask_data.device)
                    return torch.cat([general_mask_data, placeholder_mask], dim=0)
                elif (data.post_results[1] and len(data.post_results[1]) > 1 and
                      data.post_results[1][1] and len(data.post_results[1][1]) > 0 and data.post_results[1][1][
                          0] and hasattr(data.post_results[1][1][0], 'masks')):
                    # 只有夹爪分支有mask数据，创建一个与夹爪mask相同形状的零张量作为占位符
                    gripper_mask_data = data.post_results[1][1][0].masks.data
                    # 创建占位符mask，形状与夹爪mask相同但只有一行
                    placeholder_mask = torch.zeros((1, gripper_mask_data.shape[1], gripper_mask_data.shape[2]),
                                                   dtype=gripper_mask_data.dtype,
                                                   device=gripper_mask_data.device)
                    return torch.cat([placeholder_mask, gripper_mask_data], dim=0)
                else:
                    return None
            else:
                # 检查数据是否存在，避免访问None对象的属性
                if (data.post_results[1] and len(data.post_results[1]) > 0 and
                        data.post_results[1][0] and hasattr(data.post_results[1][0], 'masks')):
                    return data.post_results[1][0].masks.data
            return None
        except (TypeError, IndexError, AttributeError) as e:
            logger.warning(f"获取掩码数据时出错: {e}")
            return None

    def _process_masks(self, data, all_masks, all_detections, stable_indices):
        """
        处理掩码数据，提取有效的掩码坐标和大小。
        """
        masks_coords = []
        mask_sizes = []

        # 增强空值检查
        if all_masks is None or len(all_masks) == 0:
            logger.debug("未检测到任何掩码")
            return masks_coords, mask_sizes, np.array([])

        # 确保类型正确
        if not isinstance(all_masks, torch.Tensor):
            logger.warning("掩码数据类型不正确")
            return masks_coords, mask_sizes, np.array([])

        if len(all_masks) != len(all_detections):
            logger.warning(f"掩码和检测框数量不一致，无法对齐: 掩码数={len(all_masks)}, 检测框数={len(all_detections)}")
            return masks_coords, mask_sizes, np.array([])

        # 处理稳定索引
        if stable_indices is None or len(stable_indices) == 0:
            stable_masks = all_masks
        else:
            try:
                # 过滤掉超出范围的索引，避免CUDA索引越界错误
                valid_indices = [idx for idx in stable_indices if 0 <= idx < len(all_masks)]
                if len(valid_indices) != len(stable_indices):
                    logger.debug(f"发现无效索引，原始索引数: {len(stable_indices)}, 有效索引数: {len(valid_indices)}")

                if len(valid_indices) == 0:
                    logger.debug("没有有效的索引可用于筛选掩码")
                    return masks_coords, mask_sizes, np.array([])

                stable_masks = all_masks[valid_indices]
            except Exception as e:
                logger.error(f"根据稳定索引筛选掩码时出错: {e}")
                return masks_coords, mask_sizes, np.array([])

        masks_array = stable_masks.cpu().numpy()

        # 过滤掉全零的掩码（占位符掩码）
        valid_masks = []
        valid_indices = []
        for i, mask in enumerate(masks_array):
            if not np.all(mask == 0):  # 跳过全零掩码（占位符）
                valid_masks.append(mask)
                valid_indices.append(i)

        if not valid_masks:
            return masks_coords, mask_sizes, np.array([])

        valid_masks_array = np.array(valid_masks)

        for i, mask in enumerate(valid_masks_array):
            try:
                contours = mask_T_contour(mask)
                if not contours or len(contours) == 0:
                    continue

                areas = [cv2.contourArea(cnt) for cnt in contours]
                if not areas:
                    continue

                max_area = max(areas)
                area_threshold = max_area

                valid_contours = [
                    cnt for cnt, area in zip(contours, areas) if area >= area_threshold
                ]

                if not valid_contours:
                    continue

                combined_contour = np.vstack(valid_contours).squeeze()
                mask_sizes.append(len(combined_contour))
                masks_coords.extend(combined_contour.tolist())  # 确保转换为列表
            except Exception as e:
                logger.warning(f"处理第{i}个掩码时出错: {e}")
                continue

        return masks_coords, mask_sizes, valid_masks_array

    def _filter_object(self, data, general_masks, gripper_masks, general_masks_coords, general_mask_sizes):
        """
        根据与夹爪mask的重叠程度过滤通用分支的检测结果。
        如果通用分支mask与夹爪mask的重叠度大于阈值，则删除该通用分支的结果。

        参数:
            data: 数据对象
            general_masks: 通用分支的掩码
            gripper_masks: 夹爪掩码（只有一个）
            general_masks_coords: 通用分支掩码坐标
            general_mask_sizes: 通用分支掩码大小
        """
        # 获取需要删除的索引
        indices_to_remove = []

        # 如果没有夹爪mask，直接返回
        if len(gripper_masks) == 0:
            return general_masks_coords, general_mask_sizes

        # 获取夹爪mask（只有一个）
        gripper_mask_np = gripper_masks[0].astype(np.uint8)
        # 计算夹爪mask的非零元素数量
        gripper_count = np.count_nonzero(gripper_mask_np)

        # 如果夹爪mask为空，直接返回
        if gripper_count == 0:
            return general_masks_coords, general_mask_sizes

        # 预计算夹爪mask的边界框，用于快速筛选
        gripper_y_indices, gripper_x_indices = np.where(gripper_mask_np)
        if len(gripper_y_indices) == 0:  # 夹爪mask为空
            return general_masks_coords, general_mask_sizes

        gripper_bbox = (
            np.min(gripper_x_indices), np.min(gripper_y_indices),
            np.max(gripper_x_indices), np.max(gripper_y_indices)
        )

        # 遍历通用分支的mask
        for i in range(len(general_masks)):
            general_mask = general_masks[i].astype(np.uint8)

            # 计算通用mask的边界框
            general_y_indices, general_x_indices = np.where(general_mask)
            if len(general_y_indices) == 0:  # 通用mask为空
                continue

            general_bbox = (
                np.min(general_x_indices), np.min(general_y_indices),
                np.max(general_x_indices), np.max(general_y_indices)
            )

            # 快速边界框检查：如果两个边界框不相交，则重叠度为0
            if (general_bbox[0] > gripper_bbox[2] or general_bbox[2] < gripper_bbox[0] or
                    general_bbox[1] > gripper_bbox[3] or general_bbox[3] < gripper_bbox[1]):
                continue  # 不相交，跳过详细计算

            # 计算交集
            intersection = np.bitwise_and(general_mask, gripper_mask_np)
            intersection_count = np.count_nonzero(intersection)

            # 以通用分支mask为分母计算重叠度
            general_count = np.count_nonzero(general_mask)

            if general_count > 0:
                overlap_ratio = intersection_count / general_count
                # 如果重叠度大于0.5，则标记为需要删除
                if overlap_ratio > 0.5:
                    indices_to_remove.append(i)

        # 如果没有需要删除的项目，直接返回
        if not indices_to_remove:
            return general_masks_coords, general_mask_sizes

        # 删除跟踪结果中对应的项
        if hasattr(data, 'tracked_objects') and data.tracked_objects is not None:
            # 确保 tracked_objects 是列表形式
            if isinstance(data.tracked_objects, np.ndarray):
                tracked_objects_list = data.tracked_objects.tolist() if data.tracked_objects.size > 0 else []
            else:
                tracked_objects_list = list(data.tracked_objects) if data.tracked_objects else []

            # 从后往前删除，避免索引变化问题
            for i in sorted(indices_to_remove, reverse=True):
                if i < len(tracked_objects_list):
                    tracked_objects_list.pop(i)

            # 更新 tracked_objects
            data.tracked_objects = tracked_objects_list if tracked_objects_list else []

        # 删除mask相关数据中对应的项
        new_masks_coords = []
        new_mask_sizes = []

        current_index = 0
        for i, mask_size in enumerate(general_mask_sizes):
            # 如果当前索引不在删除列表中，则保留
            if i not in indices_to_remove:
                # 复制对应的mask坐标
                coords = general_masks_coords[current_index:current_index + mask_size]
                new_masks_coords.extend(coords)
                new_mask_sizes.append(mask_size)
            # 否则跳过（删除）
            current_index += mask_size

        return new_masks_coords, new_mask_sizes


class PipelinePredictor:
    """GPU流式流水线预测器：使用多线程队列驱动模式"""

    def __init__(self, cfg=None, overrides=None):
        self.orig_h = 640
        self.orig_w = 480
        self.stable_tracks = set()
        self.track_filter = 1
        self.frame_count = 0
        self.track_appearances = {}
        self.track_warmup = {}
        self.task = None
        self.total_frames_received = 0
        self.total_frames_processed = 0
        self.stop_flag = False
        self.cfg = cfg
        self.overrides = overrides
        self.tracking_history = {}

        # 各阶段队列（增加缓冲区大小）
        max_size = 100
        self.input_to_infer_queue = queue.Queue(maxsize=max_size)
        self.infer_to_post_queue = queue.Queue(maxsize=max_size)
        self.post_to_track_queue = queue.Queue(maxsize=max_size)
        self.track_to_result_queue = queue.Queue(maxsize=max_size)
        self.result_to_output_queue = queue.Queue(maxsize=max_size)

        # 模型初始化
        self.device = select_device('0')
        self.tensorrt_model_det = TensorRTBackend(
            weights=self.cfg['model_cfg']['model_det'],
            device=self.device,
            asynchronous=False
        )
        self.tensorrt_model_seg = TensorRTBackend(
            weights=self.cfg['model_cfg']['model_seg'],
            device=self.device,
            asynchronous=False
        )
        # self.tracker = ByteTrack(track_buffer=10, match_thresh=0.5)
        self.general_tracker = BotSort(
            reid_weights=Path("tracker/osnet_x0_25_market.pt"),  # ReID model to use
            device=torch.device("cuda:0"),
            half=True,
            frame_rate=10,
            with_reid=False,
            match_thresh=0.8,
            track_buffer=30,
            # 检测阈值
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            # CMC 选择
            cmc_method="sof",
        )

        self.gripper_tracker = BotSort(
            reid_weights=Path("tracker/osnet_x0_25_market.pt"),  # ReID model to use
            device=torch.device("cuda:0"),
            half=True,
            frame_rate=10,
            with_reid=False,
            match_thresh=0.8,
            track_buffer=30,
            # 检测阈值
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            # CMC 选择
            cmc_method="sof",
        )

        # 线程相关字段初始化为 None
        self.inference_thread = None
        self.postprocess_thread = None
        self.tracking_thread = None
        self.result_thread = None
        self.threads = []

        self.predictor = None
        self.tensorrt_model = None

    def init_predictor(self, task):
        self.task = task
        if task == 'detect':
            self.predictor = DynamicDetectionPredictor()
            self.predictor.model = self.tensorrt_model_det
            self.tensorrt_model = self.tensorrt_model_det
        elif task == 'segment':
            if self.cfg['model_cfg']['model_dual']:
                self.predictor = DualBranchPredictor()  # 双分支分割预测器
            else:
                self.predictor = DynamicSegmentationPredictor()  # 单分支分割预测器
            self.predictor.model = self.tensorrt_model_seg
            self.tensorrt_model = self.tensorrt_model_seg
        else:
            raise ValueError(f"不支持的任务类型: {task}")

        self.predictor.device = select_device('0')
        self.predictor.imgsz = 640

    def set_source(self, source_handler):
        self.source_handler = source_handler

    def start_threads(self):
        if not self.threads:
            self.inference_thread = InferenceStage(self, self.input_to_infer_queue, self.infer_to_post_queue)
            self.postprocess_thread = PostProcessStage(self, self.infer_to_post_queue, self.post_to_track_queue)
            self.tracking_thread = TrackingStage(self, self.post_to_track_queue, self.track_to_result_queue)
            self.result_thread = ResultProcessingStage(self, self.track_to_result_queue, self.result_to_output_queue)
            self.threads = [self.inference_thread, self.postprocess_thread, self.tracking_thread, self.result_thread]

        for t in self.threads:
            if not t.is_alive():
                t.start()

    def stop_threads(self):
        """停止所有流水线线程"""
        self.stop_flag = True
        for t in self.threads:
            if t.is_alive():
                t.stop()
                t.join()
        self.threads.clear()

    def stop_pipeline(self):
        self.stop_flag = True
        self.stop_threads()

        # 释放模型资源
        if hasattr(self.tensorrt_model, 'release'):
            self.tensorrt_model.release()
        torch.cuda.empty_cache()
        logger.info("Pipeline 已安全关闭")

    def run_pipeline(self):
        try:
            frame_data = self.source_handler
            if frame_data is None:
                logger.warning("无有效帧")
                return None
            frame, timestamp = frame_data
            self.orig_h, self.orig_w = frame.shape[:2]

            pre_start = time.time()
            frames = [frame]
            self.predictor.batch = ([f"frame_{timestamp}"], frames)
            im = self.predictor.preprocess(frames)
            pre_time = (time.time() - pre_start) * 1e3

            data = PipelineData(
                frame=frame.copy(),
                im=im.clone(),
                timestamp=timestamp,
                pre_time=pre_time,
            )

            # 非阻塞放入队列（增加超时处理）
            try:
                self.input_to_infer_queue.put(data, timeout=0.1)
            except queue.Full:
                logger.warning("输入队列已满，丢弃帧")
                return None

            try:
                result = self.result_to_output_queue.get(timeout=1)
                return result
            except queue.Empty:
                logger.warning("推理队列超时")
                return None

        except Exception as e:
            logger.error(f"Pipeline 执行失败: {e}")
            return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_pipeline()
        self.tensorrt_model.release()
        torch.cuda.empty_cache()
        logger.info("Pipeline 已安全关闭")
