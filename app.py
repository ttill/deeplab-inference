import mimetypes
import typer
from pathlib import Path
from mt_inference import metrics
from mt_inference.model import DeepLabModel
from mt_inference.image_util import Windowed
from mt_inference.output import write as write_output
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np


def inference(
    model, input, output, visualize_progress, visualize_result, ground_truth=None
):
    typer.echo(f"Running inference on {input}")
    windowed = Windowed(input)

    prob_map = np.zeros((windowed.height, windowed.width))

    for crop, box in windowed.window():
        crop_map = model.run(crop)

        crop_map_full = np.zeros((windowed.height, windowed.width))
        crop_map_full[box.upper : box.lower, box.left : box.right] = crop_map

        prob_map = np.maximum(prob_map, crop_map_full)

        if visualize_progress:
            vis_segmentation(windowed.image, prob_map)

    seg_map = np.round(prob_map)

    typer.echo(f"Completed running inference on {input}")

    write_output(output, seg_map, input)

    if visualize_result:
        vis_segmentation(windowed.image, prob_map, seg_map)

    if ground_truth:
        typer.echo("Ground truth provided. Calculating evaluation metrics")
        iou = metrics.IoUMetric(ground_truth, seg_map)()
        typer.echo(f"IoU: {iou}")


# we are not checking whether frozen_graph file exists (`exists=True`)
# to be able to support google cloud storage paths (gs://…)
def main(
    frozen_graph: Path = typer.Option(
        ..., file_okay=True, dir_okay=False, readable=True
    ),
    input: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=True, readable=True
    ),
    output: Path = typer.Option(None, exists=False, writable=True),
    visualize_progress: bool = False,
    visualize_result: bool = False,
    ground_truth: Path = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False, readable=True
    ),
):
    """
    Run inference on Deeplab model stored in frozen graph on image.
    Image has to be in RGB format. It will be cropped in a sliding window
    approach…
    """
    model = DeepLabModel(str(frozen_graph))

    if input.is_dir():
        for child in input.iterdir():
            if child.is_file():
                (t, enc) = mimetypes.guess_type(child)
                if t and t.startswith("image/"):
                    inference(
                        model, child, output, visualize_progress, visualize_result
                    )
    else:
        inference(
            model, input, output, visualize_progress, visualize_result, ground_truth
        )


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
