from typing import override, Optional, Tuple, Generator

import numpy as np

from src.hit_record import HitRecord
from src.color import Color
from src.materials.material import Material
from src.ray import Ray
from src.vector import random_unit_vector, Vector3


class Lambertian(Material):
    def __init__(self, rng: Generator, albedo: Color) -> None:
        self.rng = rng
        self.albedo = albedo

    @override
    def scatter(self, r_in: Ray, rec: HitRecord) -> Optional[Tuple[Color, Ray]]:
        scatter_direction: Vector3 = rec.normal + random_unit_vector(self.rng)

        if (np.abs(scatter_direction) < np.finfo(np.float64).smallest_subnormal).all():
            scatter_direction = rec.normal

        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return attenuation, scattered