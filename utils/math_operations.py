from math import pi as PI
from math import sqrt

import numpy as np
from config import FOCAL_LENGTH, MATRIX_HEIGHT, MATRIX_WIDTH
from models.point import Point2d, Point3d


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


def multiply_matrices(m1: np.array, m2: np.array):
    return np.dot(m1, m2)


def get_vector_for_points(p1: Point3d, p2: Point3d):
    return Point3d(x=p2.x - p1.x, y=p2.y - p1.y, z=p2.z - p1.z)
