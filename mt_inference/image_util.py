from collections import namedtuple
from pathlib import Path
from PIL import Image
import typer

Box = namedtuple("Box", ["left", "upper", "right", "lower"])


class Windowed:
    def __init__(self, path: Path, crop_size: int = 512, overlap: int = 50):
        self.image = Image.open(path)
        self.crop_size = crop_size
        self.stride = crop_size - overlap

    @property
    def width(self):
        return self.image.size[0]

    @property
    def height(self):
        return self.image.size[1]

    def window(self):
        box = Box(left=0, upper=0, right=self.crop_size, lower=self.crop_size)
        last_row = False

        while True:
            region = self.image.crop(box)
            typer.echo(box)
            yield region, box

            if box.right == self.width:
                box = box._replace(left=0, upper=box.upper + self.stride)
            else:
                box = box._replace(left=box.left + self.stride)

            """
            Our approach below potentially causes big overlaps
            for last row/column, but since the following could
            result in some very small crops, we would feed data
            with "wrong" scale into the model.

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
