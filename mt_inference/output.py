from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import typer


def save_image(filepath: str, seg_map: np.array):
    image = Image.fromarray((seg_map * 255).astype(np.uint8))
    with tf.io.gfile.GFile(filepath, "w") as f:
        image.save(f, "PNG")


def write(output: Path, seg_map: np.array):
    if not output is None:
        if output.is_dir():
            typer.secho(
                "Writing to dir not supported, yet.", err=True, fg=typer.colors.RED
            )
            return

        if not output.exists():
            save_image(str(output), seg_map)
        else:
            typer.secho("Output file already exists.", err=True, fg=typer.colors.RED)

    else:
        typer.secho(
            "output option not given. Segmentation map not saved.",
            fg=typer.colors.YELLOW,
        )
