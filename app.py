import typer
from pathlib import Path
from mt_inference.model import DeepLabModel
from mt_inference.image_util import Windowed
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np


# we are not checking whether frozen_graph file exists (`exists=True`)
# to be able to support google cloud storage paths (gs://…)
# TODO also support folder (dir_okay=True)
def main(
    frozen_graph: Path = typer.Option(
        ..., file_okay=True, dir_okay=False, readable=True
    ),
    image: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True
    ),
    visualize_progress: bool = False,
):
    """
    Run inference on deeplab model stored in frozen graph on image.
    Image has to be in RGB format. It will be cropped in a sliding window
    approach…
    """
    model = DeepLabModel(str(frozen_graph))

    windowed = Windowed(image)

    prob_map = np.zeros((windowed.height, windowed.width))

    for crop, box in windowed.window():
        crop_map = model.run(crop)

        crop_map_full = np.zeros((windowed.height, windowed.width))
        crop_map_full[box.upper : box.lower, box.left : box.right] = crop_map

        prob_map = np.maximum(prob_map, crop_map_full)

        if visualize_progress:
            vis_segmentation(windowed.image, prob_map)

    vis_segmentation(windowed.image, prob_map)


def vis_segmentation(image, prob_map):
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
    plt.imshow((np.round(prob_map) * 255).astype(np.uint8), alpha=0.7)
    plt.axis("off")
    plt.title("segmentation overlay")

    plt.grid("off")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
