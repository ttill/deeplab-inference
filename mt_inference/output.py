from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import typer
from .utils import respective_file


def mask2image(seg_map: np.array) -> Image:
    return Image.fromarray((seg_map * 255).astype(np.uint8))


def save_image(filepath: str, image: Image):
    with tf.io.gfile.GFile(filepath, "w") as f:
        image.save(f, "PNG")

    typer.secho(f"Saved mask to {filepath}", fg=typer.colors.GREEN)


def write(output: Path, seg_map: np.array, input: Path):
    image = mask2image(seg_map)

    output = respective_file(input, output)

    if not output.exists():
        save_image(str(output), image)
    else:
        # TODO raise instead
        typer.secho("Output file already exists.", err=True, fg=typer.colors.RED)
