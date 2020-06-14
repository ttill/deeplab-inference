from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np


class Metric(ABC):
    def __init__(self, ground_truth: Path, seg_map: np.array):
        image = Image.open(ground_truth)
        self.ground_truth = np.array(image, dtype=bool)

        self.seg_map = seg_map.astype(bool)

    @abstractmethod
    def __call__(self):
        pass


class MIoUMetric(Metric):
    def iou(self, predictions, ground_truth):
        intersection = np.count_nonzero(predictions * ground_truth)
        union = np.count_nonzero(predictions + ground_truth)

        if union:
            return intersection / union
        else:
            1

    def __call__(self):
        iou_lupine = self.iou(self.seg_map, self.ground_truth)
        iou_background = self.iou(~self.seg_map, ~self.ground_truth)

        return (iou_lupine + iou_background) / 2
