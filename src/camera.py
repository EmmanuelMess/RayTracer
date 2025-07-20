import numpy as np
from deal import post, pre, ensure
from numpy.random import Generator
from tqdm import trange

from src.color import Color
from src.hittables.hittable import Hittable
from src.image import write_image
from src.interval import Interval
from src.point import color_range, Point3
from src.ray import Ray
from src.vector import unit_vector, lerp, Vector3, random_unit_vector, is_not_zero, random_disk_vector


class Camera:
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: 0 < vfov <= 2 * np.pi)
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: np.linalg.norm(lookfrom - lookat) > 0)
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: is_not_zero(up))
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: 0 <= defocus_angle <= 2 * np.pi)
    # TODO focus_distance constraints?
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: np.isfinite(aspect_ratio) and 0 < aspect_ratio)
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: np.isfinite(height) and 0 < height)
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: 0 < samples_per_pixel)
    @pre(lambda self, rng, vfov, lookfrom, lookat, up, defocus_angle, focus_distance, aspect_ratio, height, samples_per_pixel, max_depth: 0 < max_depth)
    def __init__(self, rng: Generator, vfov: np.float64, lookfrom: Point3, lookat: Point3, up: Vector3,
                 defocus_angle: np.float64, focus_distance: np.float64, aspect_ratio: np.float64, height: np.float64,
                 samples_per_pixel: int, max_depth: np.int64) -> None:
        self.rng = rng
        self.height = height
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth
        self.lookfrom = lookfrom
        self.lookat = lookat
        self.up = up
        self.defocus_angle = defocus_angle
        self.aspect_ratio = aspect_ratio

        self.width = np.max([np.float64(1.0), np.ceil(self.height * self.aspect_ratio)])
        self.center: Point3 = self.lookfrom

        self.pixel_samples_scale = np.float64(1.0) / np.float64(self.samples_per_pixel)

        h = np.tan(vfov / 2.0)
        viewport_height = 2 * h * focus_distance
        viewport_width = viewport_height * (self.width / self.height)

        w = unit_vector(self.lookfrom - self.lookat)
        u = unit_vector(np.cross(self.up, w))
        v = np.cross(w, u)

        viewport_u: Vector3 = viewport_width * u
        viewport_v: Vector3 = viewport_height * -v

        self.pixel_delta_u = viewport_u / self.width
        self.pixel_delta_v = viewport_v / self.height

        viewport_upper_left = self.center - (focus_distance * w) - viewport_u / 2.0 - viewport_v / 2.0
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)

        defocus_radius = focus_distance * np.tan(self.defocus_angle / 2)
        self.defocus_disk_u = u * defocus_radius
        self.defocus_disk_v = v * defocus_radius


    def render(self, world: Hittable) -> None:
        image = np.empty((np.int64(self.width), np.int64(self.height), 3), dtype=np.float64)

        for j in trange(image.shape[1], desc="Rendering"):
            for i in range(image.shape[0]):
                pixel_color = np.array([0, 0, 0], dtype=np.float64)
                for _ in range(self.samples_per_pixel):
                    r = self._get_ray(i, j)
                    color = self._ray_color(r, self.max_depth, world)
                    pixel_color += color

                image[i, j] = np.clip(self.pixel_samples_scale * pixel_color, 0.0, 1.0)

        write_image(image)

    @pre(lambda self, i, _: 0 <= i <= self.width)
    @pre(lambda self, _, j: 0 <= j <= self.height)
    def _get_ray(self, i: int, j: int) -> Ray:
        """
        Construct a camera ray originating from the defocus disk and directed at a randomly
        sampled point around the pixel location i, j.
        """
        offset = self._sample_square()
        pixel_sample = (
            self.pixel00_loc + ((i + offset[0]) * self.pixel_delta_u) + ((j + offset[1]) * self.pixel_delta_v)
        )

        ray_origin = self.center if self.defocus_angle == 0 else self._defocus_disk_sample()
        ray_direction = pixel_sample - ray_origin

        return Ray(ray_origin, ray_direction)

    @post(lambda r: (-0.5 <= r).all() and (r <= 0.5).all())
    def _sample_square(self) -> Vector3:
        """
        Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        """
        return np.array([self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.5, 0.5), 0])

    # TODO check that output vector is in camera focus plane
    def _defocus_disk_sample(self) -> Vector3:
        p = random_disk_vector(self.rng)
        return self.center + p[0] * self.defocus_disk_u + p[1] * self.defocus_disk_v

    @pre(lambda self, r, depth, world: 0 <= depth)
    @post(color_range)
    def _ray_color(self, r: Ray, depth: np.int64, world: Hittable) -> Color:
        if depth == np.int64(0):
            return np.array([0, 0, 0], dtype=np.float64)

        rec = world.hit(r, Interval(False, np.float64(0.001), np.float64(np.inf)))
        if rec:
            scatter = rec.material.scatter(r, rec)

            if scatter:
                attenuation, scattered = scatter
                return attenuation * self._ray_color(scattered, depth-np.int64(1), world)

            return np.array([0, 0, 0])

        unit_direction = unit_vector(r.direction)
        a = 0.5 * (unit_direction[1] + 1.0)
        return lerp(a, np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.7, 1.0]))
