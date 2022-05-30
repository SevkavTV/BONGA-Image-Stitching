from math import cos
from math import pi as PI
from math import sqrt

import numpy as np
from config import FOCAL_LENGTH, MATRIX_HEIGHT, MATRIX_WIDTH
from models.geo import Plane, Point2d, Point3d, Vector
from sympy import Polygon


def degree_to_radian(degree: float):
    return degree * PI / 180


def convert_mm_to_m(mm: float):
    return mm / 1000


def euclidian_distance2d(p1: Point2d, p2: Point2d):
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))


def euclidian_distance3d(p1: Point3d, p2: Point3d):
    return sqrt(
        (p1.x - p2.x) * (p1.x - p2.x)
        + (p1.y - p2.y) * (p1.y - p2.y)
        + (p1.z - p2.z) * (p1.z - p2.z)
    )


def multiply_matrices(p2: np.array, p3: np.array):
    return np.dot(p2, p3)


def get_vector_by_2_points(p1: Point3d, p2: Point3d):
    return Vector(x=p2.x - p1.x, y=p2.y - p1.y, z=p2.z - p1.z)


def get_plane_by_3_points(p1: Point3d, p2: Point3d, p3: Point3d):
    x1 = p2.x - p1.x
    x2 = p3.x - p1.x
    y1 = p2.y - p1.y
    y2 = p3.y - p1.y
    z1 = p2.z - p1.z
    z2 = p3.z - p1.z

    a = y1 * z2 - z1 * y2
    b = z1 * x2 - x1 * z2
    c = x1 * y2 - y1 * x2
    d = -p1.x * a - p1.y * b - p1.z * c

    return Plane(a, b, c, d)


def get_distance_for_longtitude(latitude: float):
    rad = (latitude * PI) / 180
    return (40000 * cos(rad) / 360) * 1000


def get_intersection_between_polygons(p1: Polygon, p2: Polygon):
    pass
