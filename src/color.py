import numpy as np
from deal import pre, post, pure

Color = np.ndarray[(3,), np.float64]

color_check = lambda c: c.dtype == np.float64 and c.shape == (3,) and (0 <= c).all() and (c <= 1.0).all()

@pre(lambda r, _, __: 0 <= r <= 1)
@pre(lambda _, g, __: 0 <= g <= 1)
@pre(lambda _, __, b: 0 <= b <= 1)
@post(color_check)
@pure
def color(r: float, g: float, b: float) -> Color:
    return np.array([r, g, b], dtype=np.float64)

