import mimetypes
import typer
from pathlib import Path
from mt_inference.model import DeepLabModel
from mt_inference.inference import InferenceEngine
from mt_inference.utils import respective_file
from mt_inference.metrics import METRICS
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


# we are not checking whether frozen_graph file exists (`exists=True`)
# to be able to support google cloud storage paths (gs://…)
def main(
    frozen_graph: Path = typer.Option(
        ..., file_okay=True, dir_okay=False, readable=True
    ),
    input: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=True, readable=True
    ),
    segmentation: Path = typer.Option(None, exists=False, writable=True),
    probability: Path = typer.Option(None, exists=False, writable=True),
    visualize_progress: bool = False,
    visualize_result: bool = False,
    ground_truth: Path = typer.Option(
        None, exists=True, file_okay=True, dir_okay=True, readable=True
    ),
    split_file: Path = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False, readable=True
    ),
):
    """
    Run inference on Deeplab model stored in frozen graph on image.
    Image has to be in RGB format. It will be cropped in a sliding window
    approach…
    """
    model = DeepLabModel(str(frozen_graph))
    engine = InferenceEngine(model, vis_segmentation if visualize_progress else None,)
    overall = {"predictions": np.array([]), "ground_truths": np.array([])}

    def run_inference(image_path):
        result = engine.run(image_path)

        if segmentation:
            result.save_segmentation(respective_file(image_path, segmentation))
        else:
            typer.secho(
                "output option not given. Segmentation map not saved.",
                fg=typer.colors.YELLOW,
            )

        if probability:
            result.save_probability(respective_file(image_path, probability))

        if ground_truth:
            gt = np.array(
                Image.open(respective_file(image_path, ground_truth)), dtype=np.uint8
            )
            metrics = result.metrics(gt)
            for metric in metrics:
                typer.echo(metric)

            overall["predictions"] = np.concatenate(
                (result.segmentation_map.flatten(), overall["predictions"])
            )
            overall["ground_truths"] = np.concatenate(
                (gt.flatten(), overall["ground_truths"])
            )

        if visualize_result:
            vis_segmentation(
                result.image, result.probability_map, result.segmentation_map
            )

    split = None
    if split_file:
        with open(split_file) as f:
            split = [x.strip() for x in f]

    if input.is_dir():
        for child in input.iterdir():
            if child.is_file():
                (t, enc) = mimetypes.guess_type(child)
                if t and t.startswith("image/"):
                    if not split or child.stem in split:
                        run_inference(child)
    else:
        run_inference(input)

    if len(overall["predictions"]) > 1:
        for Metric in METRICS:
            metric = Metric(overall["ground_truths"], overall["predictions"])
            typer.echo(f"Overall {metric}")


def vis_segmentation(image, prob_map, seg_map):
    """Visualizes input image, probability map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis("off")
    plt.title("input image")

    plt.subplot(grid_spec[1])
    plt.imshow((prob_map * 255).astype(np.uint8))
    plt.axis("off")
    plt.title("probability map")

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow((seg_map * 255).astype(np.uint8), alpha=0.7)
    plt.axis("off")
    plt.title("segmentation overlay")

    plt.grid("off")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
