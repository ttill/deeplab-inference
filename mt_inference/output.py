from pathlib import Path
from PIL import Image
from matplotlib import cm
import numpy as np
import tensorflow as tf
import typer

PSEUDOCOLOR_LUT = []


def diffmap2image(diff_map: np.array) -> Image:
    image = Image.fromarray(diff_map.astype(np.uint8))

    lut = []
    for i in range(256):
        if i == 63:
            # false negative -> red
            lut.extend([255, 0, 0])
        elif i == 127:
            # false positive -> blue
            lut.extend([0, 0, 255])
        elif i == 255:
            lut.extend([255, 255, 255])
        else:
            lut.extend([0, 0, 0])

    image.putpalette(lut)
    return image


def mask2image(seg_map: np.array, pseudocolor: bool = False) -> Image:
    image = Image.fromarray((seg_map * 255).astype(np.uint8))

    if pseudocolor:
        if not PSEUDOCOLOR_LUT:
            for i in range(256):
                r, g, b, a = cm.plasma(i / 255)
                PSEUDOCOLOR_LUT.extend([int(r * 255), int(g * 255), int(b * 255)])

        image.putpalette(PSEUDOCOLOR_LUT)

    return image


def save_image(filepath: str, image: Image):
    with tf.io.gfile.GFile(filepath, "w") as f:
        image.save(f, "PNG")

    typer.secho(f"Saved mask to {filepath}", fg=typer.colors.GREEN)


def write(output: Path, image: Image):
    if not output.exists():
        save_image(str(output), image)
    else:
        # TODO raise instead
        typer.secho("Output file already exists.", err=True, fg=typer.colors.RED)
