from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from PIL import Image
import numpy as np


class Metric(ABC):
    def __init__(self, ground_truth: Path, seg_map: np.array):
        image = Image.open(ground_truth)
        self.ground_truth = np.array(image, dtype=bool)

        self.seg_map = seg_map.astype(bool)

        self._value = None

    @property
    def weight(self):
        return self.seg_map.shape[0] * self.seg_map.shape[1]

    @property
    def value(self):
        if not self._value:
            self._value = self._calculate()
        return self._value

    @abstractmethod
    def _calculate(self):
        pass

    def __str__(self):
        return str(self.value)


class MIoUMetric(Metric):
    def iou(self, predictions, ground_truth):
        intersection = np.count_nonzero(predictions * ground_truth)
        union = np.count_nonzero(predictions + ground_truth)

        if union:
            return intersection / union
        else:
            return 1

    def _calculate(self):
        iou_lupine = self.iou(self.seg_map, self.ground_truth)
        iou_background = self.iou(~self.seg_map, ~self.ground_truth)

        return (iou_lupine + iou_background) / 2


def weighted_metrics_mean(metrics: List[Metric]):
    return sum([x.value * x.weight for x in metrics]) / sum([x.weight for x in metrics])
