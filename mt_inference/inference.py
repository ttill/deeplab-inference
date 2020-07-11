from typing import Callable, List
from pathlib import Path
from PIL import Image
import numpy as np
import typer
from . import metrics
from .model import Model
from .image_util import SlidingWindow
from .output import write as write_output, mask2image, diffmap2image


class InferenceResult:
    def __init__(
        self, input_path: Path, image: Image, prob_map: np.array, threshold: int = 0.5
    ):
        self.input_path = input_path
        self.image = image
        self.probability_map = prob_map
        self.segmentation_map = (prob_map > threshold).astype(np.uint8)

    def save_segmentation(self, output: Path):
        write_output(output, mask2image(self.segmentation_map))

    def save_probability(self, output: Path):
        write_output(output, mask2image(self.probability_map, pseudocolor=True))

    def metrics(self, ground_truth: np.array) -> List[metrics.Metric]:
        return [x(ground_truth, self.segmentation_map) for x in metrics.METRICS]

    def save_difference_map(self, output: Path, ground_truth: np.array):
        inv = (~(self.segmentation_map).astype(bool)).astype(np.uint8)

        tp = self.segmentation_map * ground_truth
        tn = inv * (~(ground_truth).astype(bool)).astype(np.uint8)
        fp = self.segmentation_map - tp
        fn = inv - tn

        # use rgb array instead by joining maps using np.stack((…),axis=2)
        diff_map = tp * 255 + fp * 127 + fn * 63

        write_output(output, diffmap2image(diff_map))


class InferenceEngine:
    def __init__(
        self,
        model: Model,
        visualize_progress: Callable[[Image.Image, np.array, np.array], None] = None,
    ):
        self.model = model
        self.visualize_progress = visualize_progress

    def run(self, input: Path) -> InferenceResult:
        typer.echo(f"Running inference on {input}")
        window = SlidingWindow(input)

        prob_map = np.zeros((window.height, window.width))

        for crop, box in window:
            crop_map = self.model.run(crop)

            crop_map_full = np.zeros((window.height, window.width))
            crop_map_full[box.upper : box.lower, box.left : box.right] = crop_map

            prob_map = np.maximum(prob_map, crop_map_full)

            if self.visualize_progress:
                self.visualize_progress(window.image, prob_map)

        typer.echo(f"Completed running inference on {input}")

        return InferenceResult(input, window.image, prob_map)
