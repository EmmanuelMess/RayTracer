from typing import override, Optional

import numpy as np
from deal import pre

from src.hit_record import HitRecord
from src.hittables.hittable import Hittable
from src.interval import Interval
from src.materials.material import Material
from src.point import Point3
from src.ray import Ray
from src.vector import Vector3, length_squared


class Sphere(Hittable):
    @pre(lambda self, center, radius, material: 0 < radius)
    def __init__(self, center: Point3, radius: np.float64, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    @override
    def hit(self, r: Ray, ray_t: Interval) -> Optional[HitRecord]:
        oc: Vector3 = self.center - r.origin
        a = length_squared(r.direction)
        h = np.dot(r.direction, oc)
        c = length_squared(oc) - self.radius ** 2

        d = h ** 2 - a * c
        if d < 0:
            return None

        sqrtd = np.sqrt(d)

        root0 = (h - sqrtd) / a
        if ray_t.surrounds(root0):
            p = r.at(root0)
            outward_normal = (p - self.center) / self.radius
            return HitRecord(p, root0, r, outward_normal, self.material)

        root1 = (h + sqrtd) / a
        if ray_t.surrounds(root1):
            p = r.at(root1)
            outward_normal = (p - self.center) / self.radius
            return HitRecord(p, root0, r, outward_normal, self.material)

        return None
