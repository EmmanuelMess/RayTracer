from typing import Tuple

from deal import pre, post
from tqdm import trange
import deal
import numpy as np

from src.color import Color, color_check
from src.interval import Interval

@pre(color_check)
@post(color_check)
def _gamma(color: Color) -> Color:
    return np.sqrt(color)

@pre(color_check)
@post(lambda r: (0 <= r).all() and (r <= 255).all())
def _convert_color(color: Color) -> Tuple[np.int64, np.int64, np.int64]:
    color_gamma_corrected = _gamma(color)
    return np.int64(np.ceil(255 * color_gamma_corrected))


@pre(lambda image: len(image.shape) == 3)
@pre(lambda image: image.shape[2] == 3)
@pre(lambda image: (0 <= image).all() and (image <= 1).all())
def write_image(image: np.array):
    with open("image.ppm", "wb+") as file:
        width, height, _ = image.shape

        file.write(f"P3\n".encode())
        file.write(f"{width} {height}\n".encode())
        file.write(f"255\n".encode())

        for j in trange(height, desc="Write"):
            for i in range(width):
                r, g, b = _convert_color(image[i, j])

                file.write(f"{r} {g} {b}\n".encode())
