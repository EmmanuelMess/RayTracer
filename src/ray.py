import numpy as np
from deal import post, pre

from src.point import Point3
from src.vector import Vector3


class Ray:
    origin: Point3
    direction: Vector3

    @pre(lambda _, __, direction: np.linalg.norm(direction.shape) > 0)
    def __init__(self, origin: Point3, direction: Vector3):
        self.origin = origin
        self.direction = direction

    @pre(lambda _, t: np.isfinite(t).all())
    @post(lambda r: np.isfinite(r).all())
    def at(self, t: np.float64) -> Vector3:
        return self.origin + t * self.direction