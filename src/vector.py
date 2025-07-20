import deal
import numpy as np
from deal import pre, post, pure
from numpy.random import Generator

Vector3 = np.ndarray[(3,), np.float64]

is_finite = lambda v: np.isfinite(v).all()
is_normalized = lambda v: np.abs(np.linalg.norm(v) - 1.0) <= 0.001
"""
is_normalized ==> is_finite
"""
is_not_zero = lambda v: (0 < np.abs(v)).any()

@pre(lambda x, _, __: np.isfinite(x))
@pre(lambda _, t, __: np.isfinite(t))
@pre(lambda _, __, z: np.isfinite(z))
@post(is_finite)
@pure
def vector3(x: float, y: float, z: float) -> Vector3:
    return np.array([x, y, z], dtype=np.float64)


@pre(is_finite)
@pre(lambda v: np.sum(v**2) != 0)
@post(is_finite)
@pure
def unit_vector(v: Vector3) -> Vector3:
    return v / np.linalg.norm(v)

@pre(lambda t, _, __: 0.0 <= t <= 1.0)
@pre(lambda _, a, __: is_finite(a))
@pre(lambda _, __, b: is_finite(b))
@post(is_finite)
@pure
def lerp(t: np.float64, a: Vector3, b: Vector3) -> Vector3:
    return (1.0-t) * a + t * b

@pre(is_finite)
@post(is_finite)
@pure
def length_squared(v: Vector3) -> np.float64:
    return np.dot(v, v)

@post(lambda r: (0.0 <= r).all() and (r <= 1.0).all())
@post(is_finite)
def random(rng: Generator) -> Vector3:
    x = rng.uniform(0.0, 1.0)
    y = rng.uniform(0.0, 1.0)
    z = rng.uniform(0.0, 1.0)
    return np.array([x, y, z], dtype=np.float64)

@post(is_normalized)
def random_unit_vector(rng: Generator) -> Vector3:
    # This is the smallest positive value d so that x / d * d is not NaN for all finite float64 x.
    smallest_normalizable = np.sqrt(np.finfo(np.float64).smallest_subnormal)
    s = rng.normal(smallest_normalizable, 1, 3)
    return s / np.linalg.norm(s)

@post(is_normalized)
@post(lambda v: v[2] == 0)
def random_disk_vector(rng: Generator) -> Vector3:
    # This is the smallest positive value d so that x / d * d is not NaN for all finite float64 x.
    smallest_normalizable = np.sqrt(np.finfo(np.float64).smallest_subnormal)
    s = rng.normal(smallest_normalizable, 1, 2)
    vector = s / np.linalg.norm(s)
    return np.array([vector[0], vector[1], 0], dtype=np.float64)

@pre(is_normalized)
@post(is_normalized)
def random_on_hemisphere(rng: Generator, normal: Vector3) -> Vector3:
    on_unit_sphere = random_unit_vector(rng)
    if np.dot(on_unit_sphere, normal) > 0.0:
        return on_unit_sphere
    else:
        return -on_unit_sphere

@pre(lambda v, _: is_finite(v))
@pre(lambda v, _: is_not_zero(v))
@pre(lambda _, n: is_normalized(n))
@post(is_finite)
@post(lambda r: (0 < np.abs(r)).any())
@pure
def reflect(v: Vector3, n: Vector3) -> Vector3:
    return v - 2 * np.dot(v, n) * n

@pre(lambda uv, _, __: is_normalized(uv))
@pre(lambda _, n, __: is_normalized(n))
@pure
def refract(uv: Vector3, n: Vector3, etai_over_etat: np.float64) -> Vector3:
    cos_theta = np.min([np.dot(-uv, n), np.float64(1.0)])
    r_out_perpendicular = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -1 * np.sqrt(np.abs(1.0 - length_squared(r_out_perpendicular))) * n
    return r_out_perpendicular + r_out_parallel

@pre(lambda cosine, _: -1.0 <= cosine <= 1.0)
@pure
def reflectance(cosine: np.float64, refraction_index: np.float64) -> np.float64:
    r0 = ((np.float64(1.0) - refraction_index) / (np.float64(1.0) + refraction_index)) ** 2
    return r0 + (1-r0) * np.pow(1 - cosine, 5)