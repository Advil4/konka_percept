import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from boxmot import BotSort, StrongSort, OcSort

from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.utils.torch_utils import select_device
from utils.dual_branch_predictor import DualBranchPredictor
from utils.dynamic_predictor import DynamicDetectionPredictor, DynamicSegmentationPredictor
from utils.helpers import mask_T_contour
from utils.logger import SingletonLogger
from utils.tensorrt_backend import TensorRTBackend

logger = SingletonLogger(
    name='pipeline',
    log_file='logs/pipeline.log',
    level='INFO',
    console=True
)


@dataclass
class PipelineData:
    """统一的数据结构用于在各阶段之间传递"""
    frame: np.ndarray
    im: torch.Tensor
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


class PipelinePredictor:
    """同步流水线预测器：直接串联执行，避免队列导致的延迟和漂移问题"""

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

        # 模型初始化
        self.device = select_device('0')
        self.tensorrt_model_seg = TensorRTBackend(
            weights=self.cfg['model_cfg']['model_seg'],
            device=self.device,
            asynchronous=False
        )

        # 追踪器初始化
        self.general_tracker = OcSort(
            min_hits=0,
            det_thresh=0.1,
            min_conf=0.1,
            max_age=10,
            delta_t=3,
            asso_threshold=0.3,
            asso_func="iou",
            use_byte=False
        )

        self.gripper_tracker = OcSort(
            min_hits=0,
            det_thresh=0.1,
            min_conf=0.1,
            max_age=10,
            delta_t=3,
            asso_threshold=0.3,
            asso_func="iou",
            use_byte=False
        )

        # 追踪器时间戳跟踪
        self.last_general_timestamp = 0
        self.last_gripper_timestamp = 0

        self.predictor = None
        self.tensorrt_model = None

    def init_predictor(self, task):
        self.task = task
        if task == 'segment':
            if self.cfg['model_cfg']['model_dual']:
                self.predictor = DualBranchPredictor()
            else:
                self.predictor = DynamicSegmentationPredictor()
            self.predictor.model = self.tensorrt_model_seg
            self.tensorrt_model = self.tensorrt_model_seg
        else:
            raise ValueError(f"不支持的任务类型: {task}")

        self.predictor.device = select_device('0')
        self.predictor.imgsz = 640

    def set_source(self, source_handler):
        self.source_handler = source_handler

    def start_threads(self):
        """保留接口，但无需启动线程"""
        pass

    def stop_threads(self):
        """保留接口"""
        pass

    def stop_pipeline(self):
        self.stop_flag = True
        if hasattr(self.tensorrt_model, 'release'):
            self.tensorrt_model.release()
        torch.cuda.empty_cache()
        logger.info("Pipeline 已安全关闭")

    def run_pipeline(self):
        """同步执行完整的检测-跟踪-结果处理流程"""
        try:
            frame_data = self.source_handler
            if frame_data is None:
                logger.warning("无有效帧")
                return None

            frame, timestamp = frame_data
            self.orig_h, self.orig_w = frame.shape[:2]

            # ==================== 阶段 1: 预处理 ====================
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

            # ==================== 阶段 2: 推理 ====================
            infer_start = time.time()
            safe_input = data.im.clone().to(self.device)
            preds = self.tensorrt_model(safe_input)
            data.infer_time = (time.time() - infer_start) * 1e3
            data.infer_results = preds

            # ==================== 阶段 3: 后处理 ====================
            post_start = time.time()
            predictor = self.predictor
            predictor.args.retina_masks = True
            predictor.args.agnostic_nms = True
            predictor.args.iou = self.cfg['model_cfg']['iou_threshold']
            predictor.args.conf = self.cfg['model_cfg']['conf_threshold']

            infer_results = data.infer_results

            if isinstance(infer_results, (list, tuple)) and len(infer_results) > 4:
                # 双分支模型
                general_detect, general_segment, gripper_detect, gripper_segment, feature_list = infer_results
                feat_20, feat_40, feat_80 = feature_list
                feature_maps = [feat_80, feat_40, feat_20]
                predictor._feats = feature_maps

                import itertools
                items_list = list(predictor.model.names.items())
                n_items = len(items_list)
                all_names_dict = dict(items_list)
                original_names = predictor.model.names

                try:
                    predictor.model.names = dict(itertools.islice(all_names_dict.items(), n_items - 1))
                    general_results = predictor.postprocess([general_detect, general_segment], data.im, [data.frame])
                except Exception as e:
                    logger.warning(f"通用分支后处理失败: {e}")
                    general_results = None

                try:
                    predictor.model.names = dict(itertools.islice(all_names_dict.items(), n_items - 1, n_items))
                    gripper_results = predictor.postprocess([gripper_detect, gripper_segment], data.im, [data.frame])
                except Exception as e:
                    logger.warning(f"夹爪分支后处理失败: {e}")
                    gripper_results = None

                predictor.model.names = original_names

                if gripper_results and len(gripper_results) > 0:
                    gripper_result = gripper_results[0]
                    if hasattr(gripper_result, 'cls') and gripper_result.cls is not None:
                        gripper_result.cls.fill_(n_items - 1)
                    if hasattr(gripper_result, 'boxes') and gripper_result.boxes is not None:
                        if hasattr(gripper_result.boxes, 'cls') and gripper_result.boxes.cls is not None:
                            gripper_result.boxes.cls.fill_(n_items - 1)

                results = [general_results, gripper_results]
                data.post_results = [2, results]
            else:
                results = predictor.postprocess(infer_results, data.im, [data.frame])
                data.post_results = [1, results]

            data.post_time = (time.time() - post_start) * 1e3

            # ==================== 阶段 4: 跟踪 ====================
            track_start = time.time()

            if not data.post_results or len(data.post_results) == 0:
                data.all_tracked_objects = []
            elif data.post_results[0] >= 2:
                # 双分支跟踪
                general_tracked_objects, gripper_tracked_objects = self._track_dual_branch_sync(data)
                if len(general_tracked_objects) > 0 and len(gripper_tracked_objects) > 0:
                    data.all_tracked_objects = general_tracked_objects + gripper_tracked_objects
                elif len(general_tracked_objects) > 0:
                    placeholder_obj = [0, 0, 0, 0, -1, 0, 0, 0, -1, []]
                    data.all_tracked_objects = general_tracked_objects + [placeholder_obj]
                elif len(gripper_tracked_objects) > 0:
                    placeholder_obj = [0, 0, 0, 0, -1, 0, 0, 0, -1, []]
                    data.all_tracked_objects = [placeholder_obj] + gripper_tracked_objects
                else:
                    data.all_tracked_objects = []
            else:
                boxes = data.post_results[1][0].boxes.cpu().numpy()
                xyxy = boxes.xyxy.copy()
                xyxy[:, 0] = np.clip(xyxy[:, 0], 0, self.orig_w)
                xyxy[:, 1] = np.clip(xyxy[:, 1], 0, self.orig_h)
                xyxy[:, 2] = np.clip(xyxy[:, 2], 0, self.orig_w)
                xyxy[:, 3] = np.clip(xyxy[:, 3], 0, self.orig_h)

                detections = np.hstack([
                    xyxy,
                    boxes.conf.reshape(-1, 1),
                    boxes.cls.reshape(-1, 1),
                ])
                data.detections = detections
                tracked_objects_old = self.general_tracker.update(detections, data.frame)
                updated_tracked_objects = []
                for obj in tracked_objects_old:
                    detection_idx = int(obj[7])
                    if 0 <= detection_idx < len(detections):
                        new_bbox = detections[detection_idx][:4]
                    else:
                        new_bbox = obj[:4]

                    if np.isnan(new_bbox).any() or new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
                        continue

                    updated_obj = obj.copy()
                    updated_obj[:4] = new_bbox
                    updated_tracked_objects.append(updated_obj)
                data.all_tracked_objects = np.array(updated_tracked_objects)

            data.track_time = (time.time() - track_start) * 1e3

            # ==================== 阶段 5: 结果处理 ====================
            result_start = time.time()
            all_detections = data.detections
            all_masks = self._get_masks_from_data(data)

            if data.post_results[0] >= 2:
                general_stable_detections, general_stable_indices = self._filter_stable_objects(
                    data.all_tracked_objects[:-1], int(data.timestamp))
                gripper_stable_detections, gripper_stable_indices = self._filter_stable_objects(
                    data.all_tracked_objects[-1:], int(data.timestamp))

                general_tracked_objects = self._build_tracked_objects(general_stable_detections, int(data.timestamp))
                gripper_tracked_objects = self._build_tracked_objects(gripper_stable_detections, int(data.timestamp))

                if general_tracked_objects and gripper_tracked_objects:
                    data.tracked_objects = general_tracked_objects + gripper_tracked_objects
                elif general_tracked_objects:
                    data.tracked_objects = general_tracked_objects
                elif gripper_tracked_objects:
                    data.tracked_objects = gripper_tracked_objects
                else:
                    data.tracked_objects = []

                if self.task == 'segment':
                    general_masks_coords, general_mask_sizes, general_masks = self._process_masks(
                        data, all_masks[:-1, :] if all_masks is not None else None,
                        all_detections[:-1, :] if all_detections is not None else None,
                        general_stable_indices)

                    if gripper_tracked_objects:
                        gripper_masks_coords, gripper_mask_sizes, gripper_masks = self._process_masks(
                            data, all_masks[-1:, :] if all_masks is not None else None,
                            all_detections[-1:, :] if all_detections is not None else None,
                            gripper_stable_indices)
                        general_masks_coords, general_mask_sizes = self._filter_object(
                            data, general_masks, gripper_masks, general_masks_coords, general_mask_sizes)
                        data.masks_coords = general_masks_coords + gripper_masks_coords
                        data.mask_sizes = general_mask_sizes + gripper_mask_sizes
                    else:
                        data.masks_coords = general_masks_coords
                        data.mask_sizes = general_mask_sizes
            else:
                stable_detections, stable_indices = self._filter_stable_objects(
                    data.all_tracked_objects[-1:, :] if data.all_tracked_objects is not None else None,
                    int(data.timestamp))
                data.tracked_objects = self._build_tracked_objects(stable_detections, int(data.timestamp))

                if self.task == 'segment':
                    masks_coords, mask_sizes = self._process_masks(data, all_masks, all_detections, stable_indices)
                    data.masks_coords = masks_coords
                    data.mask_sizes = mask_sizes

            data.result_time = (time.time() - result_start) * 1e3

            # 更新帧计数
            self.frame_count += 1

            # 清理旧追踪状态（每500帧）
            if self.frame_count % 500 == 0:
                try:
                    for tracker in [self.general_tracker, self.gripper_tracker]:
                        if tracker is not None:
                            if hasattr(tracker, 'removed_stracks'):
                                tracker.removed_stracks.clear()
                except Exception as e:
                    logger.warning(f"清理追踪器底层缓存失败: {e}")

            # 清理临时数据
            data.im = None
            data.frame = None
            data.infer_results = None
            data.post_results = None
            data.all_tracked_objects = None
            data.detections = None

            return data

        except Exception as e:
            logger.error(f"Pipeline 执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _track_dual_branch_sync(self, data):
        """同步版本的双分支跟踪"""
        try:
            general_post_results = data.post_results[1][0]
            general_boxes_result = general_post_results[0]

            if hasattr(general_boxes_result, 'boxes'):
                general_boxes = general_boxes_result.boxes
                if hasattr(general_boxes, 'cpu'):
                    general_boxes_np = general_boxes.cpu().numpy()
                elif hasattr(general_boxes, 'numpy'):
                    general_boxes_np = general_boxes.numpy()
                else:
                    general_boxes_np = np.array(general_boxes)
            else:
                general_boxes_np = np.array(general_boxes_result)

            if hasattr(general_boxes_result, 'feats') and general_boxes_result.feats is not None:
                general_feats = general_boxes_result.feats[0]
                general_feats_idx = general_boxes_result.feats[1]
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
            general_tracked_objects_old = self.general_tracker.update(general_detections, data.frame)
            updated_general_tracked_objects = []
            for obj in general_tracked_objects_old:
                detection_idx = int(obj[7])

                if 0 <= detection_idx < len(general_detections):
                    new_bbox = general_detections[detection_idx][:4]
                    new_feats = general_feats[detection_idx].tolist() if general_feats is not None else []
                    new_feats_idx = int(general_feats_idx[detection_idx]) if general_feats_idx is not None else -1
                else:
                    new_bbox = obj[:4]  # 用追踪器的预测框
                    new_feats = []
                    new_feats_idx = -1

                if np.isnan(new_bbox).any() or new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
                    continue

                updated_obj = obj.copy().tolist() if isinstance(obj, np.ndarray) else list(obj)
                updated_obj[:4] = new_bbox
                updated_obj.append(new_feats_idx)
                updated_obj.append(new_feats)
                updated_general_tracked_objects.append(updated_obj)
            general_tracked_objects = updated_general_tracked_objects

            # 夹爪分支
            gripper_post_results = data.post_results[1][1]
            if gripper_post_results and len(gripper_post_results) > 0 and gripper_post_results[0] is not None:
                gripper_boxes_result = gripper_post_results[0]
                if hasattr(gripper_boxes_result, 'boxes'):
                    gripper_boxes = gripper_boxes_result.boxes
                    if hasattr(gripper_boxes, 'cpu'):
                        gripper_boxes_np = gripper_boxes.cpu().numpy()
                    elif hasattr(gripper_boxes, 'numpy'):
                        gripper_boxes_np = gripper_boxes.numpy()
                    else:
                        gripper_boxes_np = np.array(gripper_boxes)
                else:
                    gripper_boxes_np = np.array(gripper_boxes_result)

                if hasattr(gripper_boxes_result, 'feats') and gripper_boxes_result.feats is not None:
                    gripper_feats = gripper_boxes_result.feats[0]
                    gripper_feats_idx = gripper_boxes_result.feats[1]
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
                gripper_tracked_objects_old = self.gripper_tracker.update(gripper_detections, data.frame)
                updated_gripper_tracked_objects = []
                for obj in gripper_tracked_objects_old:
                    detection_idx = int(obj[7])

                    if 0 <= detection_idx < len(gripper_detections):
                        new_bbox = gripper_detections[detection_idx][:4]
                        new_feats = gripper_feats[detection_idx].tolist() if gripper_feats is not None else []
                        new_feats_idx = int(gripper_feats_idx[detection_idx]) if gripper_feats_idx is not None else -1
                    else:
                        new_bbox = obj[:4]
                        new_feats = []
                        new_feats_idx = -1

                    if np.isnan(new_bbox).any() or new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
                        continue

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
                placeholder = np.zeros((1, general_detections.shape[1]), dtype=np.float32)
                data.detections = np.vstack([general_detections, placeholder]).astype(np.float32)
            elif gripper_detections.size > 0:
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
        """创建检测框数据（与原版相同）"""
        try:
            if hasattr(boxes_data, 'cpu'):
                boxes_np = boxes_data.cpu().numpy()
            elif hasattr(boxes_data, 'numpy'):
                boxes_np = boxes_data.numpy()
            else:
                boxes_np = np.array(boxes_data)

            if hasattr(boxes_data, 'xyxy'):
                if hasattr(boxes_data.xyxy, 'cpu'):
                    xyxy = boxes_data.xyxy.cpu().numpy().copy()
                else:
                    xyxy = boxes_data.xyxy.copy()
            else:
                xyxy = boxes_np.copy()
                if xyxy.ndim == 2 and xyxy.shape[1] >= 4:
                    xyxy = xyxy[:, :4]

            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, self.orig_w)
            xyxy[:, 1] = np.clip(xyxy[:, 1], 0, self.orig_h)
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, self.orig_w)
            xyxy[:, 3] = np.clip(xyxy[:, 3], 0, self.orig_h)

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
                if boxes_np.ndim == 2 and boxes_np.shape[1] >= 6:
                    conf_data = boxes_np[:, 4].reshape(-1, 1)
                    cls_data = boxes_np[:, 5].reshape(-1, 1)
                else:
                    conf_data = np.ones((xyxy.shape[0], 1))
                    cls_data = np.zeros((xyxy.shape[0], 1))

            detections = np.hstack([xyxy, conf_data, cls_data])
            return detections.astype(np.float32)

        except Exception as e:
            logger.error(f"创建detections时出错: {e}")
            return np.array([])

    def _filter_stable_objects(self, all_tracked_objects, timestamp):
        """筛选稳定目标（与原版相同）"""
        if not all_tracked_objects:
            return [], []

        stable_detections = []
        stable_indices = []

        for obj in all_tracked_objects:
            if isinstance(obj, (list, np.ndarray)) and len(obj) > 4:
                try:
                    track_id = int(obj[4])
                    if track_id >= 0:
                        stable_detections.append(obj)
                        stable_indices.append(int(obj[7]))
                except (IndexError, ValueError, TypeError):
                    continue

        return stable_detections, stable_indices

    def _build_tracked_objects(self, stable_detections, timestamp):
        """构建跟踪对象（与原版相同）"""
        if not stable_detections:
            return []

        tracked_objects = []
        for item in stable_detections:
            if isinstance(item, (list, np.ndarray)) and len(item) > 4:
                track_id = int(item[4]) if isinstance(item[4], (int, float)) else -1
                if track_id < 0:
                    continue

                x1, y1, x2, y2, tid, conf, cls_idx, idx, feats_idx, feats = item
                x1_clamped = max(0, min(x1, self.orig_w))
                y1_clamped = max(0, min(y1, self.orig_h))
                x2_clamped = max(0, min(x2, self.orig_w))
                y2_clamped = max(0, min(y2, self.orig_h))

                track_warmup_info = self.track_warmup.get(tid, f"{cls_idx}_{timestamp}")
                track_id_str = f"{track_warmup_info.split('_')[-1]}_{int(tid)}"

                tracked_objects.append((
                    x1_clamped, y1_clamped, x2_clamped, y2_clamped,
                    track_id_str, conf, cls_idx, idx, feats_idx, feats
                ))

        return tracked_objects

    def _get_masks_from_data(self, data):
        """获取掩码数据（与原版相同）"""
        try:
            if data.post_results[0] >= 2:
                if (data.post_results[1] and len(data.post_results[1]) > 1 and
                        data.post_results[1][0] and len(data.post_results[1][0]) > 0 and
                        data.post_results[1][0][0] and hasattr(data.post_results[1][0][0], 'masks') and
                        data.post_results[1][1] and len(data.post_results[1][1]) > 0 and
                        data.post_results[1][1][0] and hasattr(data.post_results[1][1][0], 'masks')):
                    return torch.cat([data.post_results[1][0][0].masks.data,
                                      data.post_results[1][1][0].masks.data], dim=0)
                elif (data.post_results[1] and len(data.post_results[1]) > 0 and
                      data.post_results[1][0] and len(data.post_results[1][0]) > 0 and
                      data.post_results[1][0][0] and hasattr(data.post_results[1][0][0], 'masks')):
                    general_mask_data = data.post_results[1][0][0].masks.data
                    placeholder_mask = torch.zeros((1, general_mask_data.shape[1], general_mask_data.shape[2]),
                                                   dtype=general_mask_data.dtype, device=general_mask_data.device)
                    return torch.cat([general_mask_data, placeholder_mask], dim=0)
                elif (data.post_results[1] and len(data.post_results[1]) > 1 and
                      data.post_results[1][1] and len(data.post_results[1][1]) > 0 and
                      data.post_results[1][1][0] and hasattr(data.post_results[1][1][0], 'masks')):
                    gripper_mask_data = data.post_results[1][1][0].masks.data
                    placeholder_mask = torch.zeros((1, gripper_mask_data.shape[1], gripper_mask_data.shape[2]),
                                                   dtype=gripper_mask_data.dtype, device=gripper_mask_data.device)
                    return torch.cat([placeholder_mask, gripper_mask_data], dim=0)
                else:
                    return None
            else:
                if (data.post_results[1] and len(data.post_results[1]) > 0 and
                        data.post_results[1][0] and hasattr(data.post_results[1][0], 'masks')):
                    return data.post_results[1][0].masks.data
            return None
        except (TypeError, IndexError, AttributeError) as e:
            logger.warning(f"获取掩码数据时出错: {e}")
            return None

    def _process_masks(self, data, all_masks, all_detections, stable_indices):
        masks_coords = []
        mask_sizes = []
        valid_masks_list = []

        # 如果输入为空，返回全0数组以保证长度对齐
        if all_masks is None or len(all_masks) == 0 or stable_indices is None or len(stable_indices) == 0:
            length = len(stable_indices) if stable_indices else 0
            return [], [0] * length, np.array([])

        for idx in stable_indices:
            # 防御性索引，遇到无效索引或全零掩码，强行补 0
            if idx < 0 or idx >= len(all_masks):
                mask_sizes.append(0)
                valid_masks_list.append(np.zeros_like(all_masks[0].cpu().numpy()))
                continue

            mask = all_masks[idx].cpu().numpy()
            if np.all(mask == 0):
                mask_sizes.append(0)
                valid_masks_list.append(mask)
                continue

            try:
                contours = mask_T_contour(mask)
                if not contours or len(contours) == 0:
                    mask_sizes.append(0)
                    valid_masks_list.append(mask)
                    continue

                areas = [cv2.contourArea(cnt) for cnt in contours]
                if not areas:
                    mask_sizes.append(0)
                    valid_masks_list.append(mask)
                    continue

                max_area = max(areas)
                valid_contours = [cnt for cnt, area in zip(contours, areas) if area >= max_area]

                if not valid_contours:
                    mask_sizes.append(0)
                    valid_masks_list.append(mask)
                    continue

                combined_contour = np.vstack(valid_contours).reshape(-1, 2)
                mask_sizes.append(len(combined_contour))
                masks_coords.extend(combined_contour.tolist())
                valid_masks_list.append(mask)
            except Exception as e:
                logger.warning(f"处理第{idx}个掩码时出错: {e}")
                mask_sizes.append(0)
                valid_masks_list.append(mask)

        return masks_coords, mask_sizes, np.array(valid_masks_list)

    def _filter_object(self, data, general_masks, gripper_masks, general_masks_coords, general_mask_sizes):
        """过滤重叠对象（与原版相同）"""
        if len(gripper_masks) == 0 or len(general_masks) == 0:
            return general_masks_coords, general_mask_sizes

        gripper_mask_np = gripper_masks[0].astype(bool)
        if not gripper_mask_np.any():
            return general_masks_coords, general_mask_sizes

        general_masks_bool = general_masks.astype(bool)
        intersections = np.logical_and(general_masks_bool, gripper_mask_np)
        intersection_counts = intersections.sum(axis=(1, 2))
        general_counts = general_masks_bool.sum(axis=(1, 2))

        overlap_ratios = np.zeros_like(general_counts, dtype=np.float32)
        valid_masks_idx = general_counts > 0
        overlap_ratios[valid_masks_idx] = intersection_counts[valid_masks_idx] / general_counts[valid_masks_idx]

        indices_to_remove = np.where(overlap_ratios > 0.5)[0].tolist()

        if not indices_to_remove:
            return general_masks_coords, general_mask_sizes

        if hasattr(data, 'tracked_objects') and data.tracked_objects is not None:
            if isinstance(data.tracked_objects, np.ndarray):
                tracked_objects_list = data.tracked_objects.tolist() if data.tracked_objects.size > 0 else []
            else:
                tracked_objects_list = list(data.tracked_objects) if data.tracked_objects else []

            for i in sorted(indices_to_remove, reverse=True):
                if i < len(tracked_objects_list):
                    tracked_objects_list.pop(i)
            data.tracked_objects = tracked_objects_list if tracked_objects_list else []

        new_masks_coords = []
        new_mask_sizes = []
        current_index = 0
        for i, mask_size in enumerate(general_mask_sizes):
            if i not in indices_to_remove:
                new_masks_coords.extend(general_masks_coords[current_index:current_index + mask_size])
                new_mask_sizes.append(mask_size)
            current_index += mask_size

        return new_masks_coords, new_mask_sizes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_pipeline()
        self.tensorrt_model.release()
        torch.cuda.empty_cache()
        logger.info("Pipeline 已安全关闭")
