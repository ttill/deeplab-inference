from abc import ABC, abstractmethod
from math import sqrt
from typing import List
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


class Metric(ABC):
    NAME = None

    def __init__(self, ground_truth: np.array, seg_map: np.array):
        self.ground_truth = ground_truth
        self.seg_map = seg_map

        self._value = None

    @property
    def value(self):
        if not self._value:
            self._value = self._calculate()
        return self._value

    @abstractmethod
    def _calculate(self):
        pass

    def __str__(self):
        return f"{self.NAME}: {self.value}"


class KerasMetric(Metric):
    @abstractmethod
    def _keras_metric(self):
        pass

    def _calculate(self):
        metric = self._keras_metric()
        metric.update_state(self.ground_truth, self.seg_map)
        return metric.result().numpy()


class MIoUMetric(KerasMetric):
    NAME = "MIoU"

    def _keras_metric(self):
        return tf.keras.metrics.MeanIoU(num_classes=2)


class AccuracyMetric(KerasMetric):
    NAME = "Accuracy"

    def _keras_metric(self):
        return tf.keras.metrics.Accuracy()


class PrecisionMetric(KerasMetric):
    NAME = "Precision"

    def _keras_metric(self):
        return tf.keras.metrics.Precision()


class RecallMetric(KerasMetric):
    NAME = "Recall"

    def _keras_metric(self):
        return tf.keras.metrics.Recall()


class MatthewsCorrelationCoefficientMetric(Metric):
    NAME = "Matthews Correlation Coefficient"

    def _calculate(self):
        gt = self.ground_truth.astype(bool)
        sm = self.seg_map.astype(bool)

        tp = np.count_nonzero(sm * gt)  # true positive
        tn = np.count_nonzero(~sm * ~gt)  # true negative
        fp = np.count_nonzero(sm) - tp  # false positive
        fn = np.count_nonzero(~sm) - tn  # false negative

        numerator = tp * tn - fp * fn
        denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if not denominator:
            return 0

        return numerator / denominator


METRICS = [
    MIoUMetric,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    MatthewsCorrelationCoefficientMetric,
]
