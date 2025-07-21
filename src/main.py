import deal
import numpy as np
from numpy.random import Generator

from src.camera import Camera
from src.color import color
from src.hittables.hittable_list import HittableList
from src.hittables.sphere import Sphere
from src.materials.dielectric import Dielectric
from src.materials.lambertian import Lambertian
from src.materials.metal import Metal
from src.vector import vector3


def world1(rng: Generator) -> HittableList:
    material_ground = Lambertian(rng, color(0.8, 0.8, 0.0))
    material_center = Lambertian(rng, color(0.1, 0.2, 0.5))
    material_left = Dielectric(rng, np.float64(1.5))
    material_right = Metal(rng, color(0.8, 0.6, 0.2), np.float64(1.0))

    world = HittableList()

    world.add(Sphere(vector3( 0.0, -100.5, -1.0), np.float64(100.0), material_ground))
    world.add(Sphere(vector3( 0.0,    0.0, -1.2), np.float64(0.5), material_center))
    world.add(Sphere(vector3(-1.0,    0.0, -1.0), np.float64(0.5), material_left))
    world.add(Sphere(vector3( 1.0,    0.0, -1.0), np.float64(0.5), material_right))

    return world


def world2(rng: Generator) -> HittableList:
    R = np.cos(np.pi / 4)

    material_left = Lambertian(rng, color(0.0, 0.0, 1.0))
    material_right = Lambertian(rng, color(1.0, 0.0, 0.0))

    world = HittableList()
    world.add(Sphere(vector3(-R, 0.0, -1.0), R, material_left))
    world.add(Sphere(vector3( R, 0.0, -1.0), R, material_right))

    return world

def world3(rng: Generator) -> HittableList:
    world = HittableList()

    ground_material = Lambertian(rng, color(0.5, 0.5, 0.5))
    ground_sphere = Sphere(vector3(0, -1000, 0.0), np.float64(1000), ground_material)

    world.add(ground_sphere)

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_material = rng.choice(["diffuse", "metal", "glass"], p=[0.8, 0.15, 0.05])
            center_x = a + 0.9*rng.uniform(0.0, 1.0)
            center_y = 0.2
            center_z = b + 0.9*rng.uniform(0.0, 1.0)

            center = vector3(center_x, center_y, center_z)

            if np.linalg.norm(center - vector3(4, 0.2, 0)) > 0.9:
                sphere_material = None

                if choose_material == "diffuse":
                    albedo = rng.uniform(0.0, 1.0, 3) * rng.uniform(0.0, 1.0, 3)
                    sphere_material = Lambertian(rng, albedo)
                elif choose_material == "metal":
                    albedo = rng.uniform(0.5, 1.0, 3)
                    fuzz = np.float64(rng.uniform(0, 0.5))
                    sphere_material = Metal(rng, albedo, fuzz)
                elif choose_material == "glass":
                    sphere_material = Dielectric(rng, np.float64(1.5))

                assert sphere_material is not None

                sphere = Sphere(center, np.float64(0.2), sphere_material)
                world.add(sphere)

    material1 = Dielectric(rng, np.float64(1.5))
    sphere1 = Sphere(vector3(0, 1, 0), np.float64(1.0), material1)
    world.add(sphere1)

    material2 = Lambertian(rng, color(0.4, 0.2, 0.1))
    sphere2 = Sphere(vector3(-4, 1, 0), np.float64(1.0), material2)
    world.add(sphere2)

    material3 = Metal(rng, color(0.7, 0.6, 0.5), np.float64(0.0))
    sphere3 = Sphere(vector3(0.7, 0.6, 0.5), np.float64(1.0), material3)
    world.add(sphere3)

    return world


def camera1(rng: Generator) -> Camera:
    vfov = np.deg2rad(90)
    lookfrom = vector3(-2, 2, 1)
    lookat = vector3(0, 0, -1)
    up = vector3(0, 1, 0)
    defocus_angle = np.deg2rad(np.float64(10.0))
    focus_dist = np.float64(3.4)
    aspect_ratio = np.float64(16.0) / np.float64(9.0)
    height = np.float64(200)
    samples_per_pixel = np.int64(100)
    max_depth = np.int64(10)
    camera = Camera(rng, vfov, lookfrom, lookat, up, defocus_angle, focus_dist, aspect_ratio, height, samples_per_pixel, max_depth)
    return camera


def camera2(rng: Generator) -> Camera:
    vfov = np.deg2rad(20)
    lookfrom = vector3(-2, 2, 1)
    lookat = vector3(0, 0, -1)
    up = vector3(0, 1, 0)
    defocus_angle = np.deg2rad(np.float64(10.0))
    focus_dist = np.float64(3.4)
    aspect_ratio = np.float64(16.0) / np.float64(9.0)
    height = np.float64(200)
    samples_per_pixel = np.int64(100)
    max_depth = np.int64(10)
    camera = Camera(rng, vfov, lookfrom, lookat, up, defocus_angle, focus_dist, aspect_ratio, height, samples_per_pixel, max_depth)
    return camera


def camera3(rng: Generator) -> Camera:
    vfov = np.deg2rad(20)
    lookfrom = vector3(13, 2, 3)
    lookat = vector3(0, 0, 0)
    up = vector3(0, 1, 0)
    defocus_angle = np.deg2rad(np.float64(0.6))
    focus_dist = np.float64(10.0)
    aspect_ratio = np.float64(16.0 / 9.0)
    height = np.float64(1200 * (9 / 16))
    samples_per_pixel = np.int64(500)
    max_depth = np.int64(50)
    camera = Camera(rng, vfov, lookfrom, lookat, up, defocus_angle, focus_dist, aspect_ratio, height, samples_per_pixel, max_depth)
    return camera

def main():
    rng = np.random.default_rng()

    world = world1(rng)

    camera = camera1(rng)

    camera.render(world)

if __name__ == '__main__':
    main()