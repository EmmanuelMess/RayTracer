from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

from src.color import Color
from src.ray import Ray


class Material(metaclass=ABCMeta):
    @abstractmethod
    def scatter(self, r_in: Ray, rec: "HitRecord") \
            -> Optional[Tuple[Color, Ray]]:
        pass