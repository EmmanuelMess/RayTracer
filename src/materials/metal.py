from typing import override, Optional, Tuple

import numpy as np
from deal import pre
from numpy.random import Generator

from src.hit_record import HitRecord
from src.color import Color
from src.materials.material import Material
from src.ray import Ray
from src.vector import reflect, random_unit_vector, unit_vector


class Metal(Material):
    @pre(lambda self, rng, albedo, fuzz: 0.0 <= fuzz <= 1.0)
    def __init__(self, rng: Generator, albedo: Color, fuzz: np.float64) -> None:
        self.rng = rng
        self.albedo = albedo
        self.fuzz = fuzz

    @override
    def scatter(self, r_in: Ray, rec: HitRecord) -> Optional[Tuple[Color, Ray]]:
        reflected = reflect(r_in.direction, rec.normal)
        fuzzied = unit_vector(reflected) + (self.fuzz * random_unit_vector(self.rng))
        scattered = Ray(rec.p, fuzzied)
        attenuation = self.albedo

        if np.dot(scattered.direction, rec.normal) <= 0:
            return None

        return attenuation, scattered