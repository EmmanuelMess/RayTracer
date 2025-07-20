from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

from src.hit_record import HitRecord
from src.interval import Interval
from src.ray import Ray


class Hittable(metaclass=ABCMeta):
    @abstractmethod
    def hit(self, r: Ray, ray_t: Interval) -> Optional[HitRecord]:
        pass
