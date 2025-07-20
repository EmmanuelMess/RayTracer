from src.vector import Vector3

Point3 = Vector3

color_range = lambda c: (0 <= c).all() and (c <= 1.0).all()