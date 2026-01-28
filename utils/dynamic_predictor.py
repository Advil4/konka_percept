# scrips/dynamic_predictor.py

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor

class DynamicDetectionPredictor(DetectionPredictor):
    pass

class DynamicSegmentationPredictor(SegmentationPredictor):
    pass
