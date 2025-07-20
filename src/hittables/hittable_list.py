from typing import List, Optional

from numba import njit

from src.hit_record import HitRecord
from src.hittables.hittable import Hittable
from src.interval import Interval
from src.ray import Ray


class HittableList(Hittable):

    def __init__(self):
        self.objects: List[Hittable] = []

    def clear(self):
        self.objects.clear()

    def add(self, object: Hittable):
        self.objects.append(object)

    def hit(self, r: Ray, ray_t: Interval) -> Optional[HitRecord]:
        closest_so_far = ray_t.upper
        rec: Optional[HitRecord] = None

        for object in self.objects:
            temp_rec = object.hit(r, Interval(False, ray_t.lower, closest_so_far))
            if temp_rec:
                closest_so_far = temp_rec.t
                rec = temp_rec

        return rec