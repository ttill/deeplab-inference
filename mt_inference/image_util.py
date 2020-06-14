from collections import namedtuple
from pathlib import Path
from typing import Iterator, Tuple
from PIL import Image
import typer

Box = namedtuple("Box", ["left", "upper", "right", "lower"])


class SlidingWindow:
    def __init__(self, path: Path, crop_size: int = 512, overlap: int = 50):
        self.image = Image.open(path)
        self.crop_size = crop_size
        self.stride = crop_size - overlap

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def height(self) -> int:
        return self.image.size[1]

    def __iter__(self) -> Iterator[Tuple[Box, Image.Image]]:
        box = Box(
            left=0,
            upper=0,
            right=min(self.crop_size, self.width),
            lower=min(self.crop_size, self.height),
        )
        last_row = self.height <= self.crop_size

        while True:
            region = self.image.crop(box)
            typer.echo(box)
            yield region, box

            if box.right == self.width:
                box = box._replace(left=0, upper=box.upper + self.stride)
            else:
                box = box._replace(left=box.left + self.stride)

            """
            Our approach below potentially causes big overlaps for
            last row/column (and thus performance impacts), but since
            the following could result in some very small crops,
            we would feed data with "wrong" scale into the model.

            box = box._replace(
                right=min(box.left + self.crop_size, self.width),
                lower=min(box.upper + self.crop_size, self.height),
            )
            """
            box = box._replace(
                right=box.left + self.crop_size, lower=box.upper + self.crop_size
            )
            if box.right > self.width:
                box = box._replace(left=self.width - self.crop_size, right=self.width)
            if box.lower > self.height:
                if last_row:
                    break

                last_row = True
                box = box._replace(
                    upper=self.height - self.crop_size, lower=self.height
                )
