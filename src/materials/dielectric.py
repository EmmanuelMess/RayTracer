from typing import override, Optional, Tuple

import numpy as np
from deal import pre
from numpy.random import Generator

from src.hit_record import HitRecord
from src.color import Color
from src.materials.material import Material
from src.ray import Ray
from src.vector import unit_vector, refract, reflect, reflectance


class Dielectric(Material):
    @pre(lambda _, __, refraction_index: np.isfinite(refraction_index))
    @pre(lambda _, __, refraction_index: 0 < refraction_index)
    def __init__(self, rng: Generator, refraction_index: np.float64) -> None:
        self.rng = rng
        self.refraction_index = refraction_index

    @override
    def scatter(self, r_in: Ray, rec: HitRecord) -> Optional[Tuple[Color, Ray]]:
        attenuation = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        ri = np.float64(1.0) / self.refraction_index if rec.front_face else self.refraction_index

        unit_direction = unit_vector(r_in.direction)
        cos_theta = np.min([np.dot(-unit_direction, rec.normal), np.float64(1.0)])
        sin_theta = np.sqrt(np.float64(1.0) - cos_theta ** 2)

        cannot_refract = ri * sin_theta > np.float64(1.0)

        if cannot_refract or reflectance(cos_theta, ri) > self.rng.uniform(0.0, 1.0):
            direction = reflect(unit_direction, rec.normal)
        else:
            direction = refract(unit_direction, rec.normal, ri)

        scattered = Ray(rec.p, direction)

        return attenuation, scattered
