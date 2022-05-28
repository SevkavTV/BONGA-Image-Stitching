from math import cos, sin

import numpy as np


def get_yaw_transformation_matrix(yaw: float):
    yaw_transformation_matrix = np.array(
        [
            [cos(yaw), sin(yaw), 0, 0],
            [-sin(yaw), cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    return yaw_transformation_matrix


def get_roll_transformation_matrix(roll: float):
    roll_transformation_matrix = np.array(
        [
            [cos(roll), 0, -sin(roll), 0],
            [0, 1, 0, 0],
            [sin(roll), 0, cos(roll), 0],
            [0, 0, 0, 1],
        ]
    )

    return roll_transformation_matrix


def get_pitch_transformation_matrix(pitch: float):
    pitch_transformation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, cos(pitch), sin(pitch), 0],
            [0, -sin(pitch), cos(pitch), 0],
            [0, 0, 0, 1],
        ]
    )

    return pitch_transformation_matrix
