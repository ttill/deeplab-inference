from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import typer


def mask2image(seg_map: np.array) -> Image:
    return Image.fromarray((seg_map * 255).astype(np.uint8))


def save_image(filepath: str, image: Image):
    with tf.io.gfile.GFile(filepath, "w") as f:
        image.save(f, "PNG")

    typer.secho(f"Saved mask to {filepath}", fg=typer.colors.GREEN)


def write(output: Path, seg_map: np.array, input: Path):
    if not output is None:
        image = mask2image(seg_map)

        if output.is_dir():
            output = output / (input.stem + ".png")

        if not output.exists():
            if not output.suffix.lower() == ".png":
                typer.secho(
                    "Wrong file suffix for output (expected .png/.PNG)",
                    fg=typer.colors.YELLOW,
                )
            save_image(str(output), image)
        else:
            typer.secho("Output file already exists.", err=True, fg=typer.colors.RED)

    else:
        typer.secho(
            "output option not given. Segmentation map not saved.",
            fg=typer.colors.YELLOW,
        )
