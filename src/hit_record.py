from dataclasses import dataclass

import numpy as np
from deal import pre

from src.materials.material import Material
from src.point import Point3
from src.ray import Ray
from src.vector import Vector3, is_normalized


class HitRecord:
    @pre(lambda self, p, t, r, outward_normal, material: is_normalized(outward_normal))
    def __init__(self, p: Point3, t: np.float64, r: Ray, outward_normal: Vector3, material: Material):
        self.p = p
        self.t = t
        self.front_face = np.dot(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        self.material = material