from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


class Metric(ABC):
    def __init__(self, ground_truth: Path, seg_map: np.array):
        image = Image.open(ground_truth)
        self.ground_truth = np.array(image, dtype=np.uint8)

        self.seg_map = seg_map

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


class KerasMetric(Metric):
    @abstractmethod
    def _keras_metric(self):
        pass

    def _calculate(self):
        metric = self._keras_metric()
        metric.update_state(self.ground_truth, self.seg_map)
        return metric.result().numpy()


class MIoUMetric(KerasMetric):
    def _keras_metric(self):
        return tf.keras.metrics.MeanIoU(num_classes=2)


class PixelAccuracyMetric(KerasMetric):
    def _keras_metric(self):
        return tf.keras.metrics.Accuracy()


def weighted_metrics_mean(metrics: List[Metric]):
    return sum([x.value * x.weight for x in metrics]) / sum([x.weight for x in metrics])
