import cv2 as cv
import numpy as np
from config import (
    FOCAL_LENGTH,
    MATRIX_HEIGHT,
    MATRIX_WIDTH,
    NORMAL_HEIGHT,
    NORMAL_WIDTH,
)
from models.geo import Point3d

from utils.math_operations import convert_mm_to_m


def normalize_img(img):
    k1 = img.shape[0] / NORMAL_HEIGHT
    k2 = img.shape[1] / NORMAL_WIDTH
    if k1 < 1:
        k1 = 1

    if k2 < 1:
        k2 = 1

    k1 = max(k1, k2)
    transformation_matrix = np.zeros((2, 3))
    transformation_matrix[0][0] = 1 / k1
    transformation_matrix[1][1] = 1 / k1
    result = cv.warpAffine(
        img, transformation_matrix, (int(img.shape[0] / k1), int(img.shape[1] / k1))
    )
    return result


def get_camera_matrix_point_on_air(center: Point3d, i: float, j: float):
    return np.array(
        [
            center.x + convert_mm_to_m(MATRIX_WIDTH) * i,
            center.y + convert_mm_to_m(MATRIX_HEIGHT) * j,
            center.z,
            1,
        ]
    )
