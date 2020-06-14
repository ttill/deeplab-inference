from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np
from .utils import respective_file


class Metric(ABC):
    def __init__(self, ground_truth: Path, seg_map: np.array, input: Path):
        ground_truth = respective_file(input, ground_truth)
        image = Image.open(ground_truth)
        self.ground_truth = np.array(image, dtype=bool)

        self.seg_map = seg_map.astype(bool)

    @abstractmethod
    def __call__(self):
        pass


class IoUMetric(Metric):
    def __call__(self):
        intersection = np.count_nonzero(self.seg_map * self.ground_truth)
        union = np.count_nonzero(self.seg_map + self.ground_truth)

        if union:
            return intersection / float(union)
        else:
            1
