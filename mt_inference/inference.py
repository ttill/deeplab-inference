from typing import Callable, List
from pathlib import Path
from PIL import Image
import numpy as np
import typer
from . import metrics
from .model import Model
from .image_util import SlidingWindow
from .output import write as write_output


class InferenceResult:
    def __init__(
        self, input_path: Path, image: Image, prob_map: np.array, threshold: int = 0.5
    ):
        self.input_path = input_path
        self.image = image
        self.probability_map = prob_map
        self.segmentation_map = (prob_map > threshold).astype(np.uint8)

    def save_segmentation(self, output: Path):
        write_output(output, self.segmentation_map)

    def save_probability(self, output: Path):
        write_output(output, self.probability_map, pseudocolor=True)

    def metrics(self, ground_truth: np.array) -> List[metrics.Metric]:
        return [x(ground_truth, self.segmentation_map) for x in metrics.METRICS]


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
