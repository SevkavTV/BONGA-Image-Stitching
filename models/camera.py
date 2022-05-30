from typing import List

import folium
import numpy as np
from config import FOCAL_LENGTH, LATITUDE_DISTANCE, MATRIX_HEIGHT, MATRIX_WIDTH
from sympy import Point, Polygon
from utils.math_operations import (
    convert_mm_to_m,
    degree_to_radian,
    get_distance_for_longtitude,
    get_vector_by_2_points,
    multiply_matrices,
)
from utils.transformation_matrices import (
    get_pitch_transformation_matrix,
    get_roll_transformation_matrix,
    get_yaw_transformation_matrix,
)

from models.geo import Point3d
from models.ground_area import GroundArea
from models.image import Image
from models.log import Log


class Camera:
    def __init__(self, log: Log) -> None:
        self.images = self._calculate_ground_area(log._retrieve_images_info())

    def _calculate_ground_area(self, images: List[Image]):
        for image in images:
            LONGTITUDE_DISTANCE = get_distance_for_longtitude(image.location.lat)

            yaw_transformation_matrix = get_yaw_transformation_matrix(
                degree_to_radian(image.rotation.yaw)
            )
            roll_transformation_matrix = get_roll_transformation_matrix(
                degree_to_radian(-1 * image.rotation.roll)
            )
            pitch_transformation_matrix = get_pitch_transformation_matrix(
                degree_to_radian(-1 * image.rotation.pitch)
            )
            transformation_matrix = multiply_matrices(
                yaw_transformation_matrix,
                multiply_matrices(
                    roll_transformation_matrix, pitch_transformation_matrix
                ),
            )

            cross_corner_mapping = {
                "upper_left": "lower_right",
                "upper_right": "lower_left",
                "lower_left": "upper_right",
                "lower_right": "upper_left",
            }
            air_matrix_coordinates = self._get_camera_matrix_coordinates_for_image(
                Point3d(0, 0, 0)
            )
            ground_image_coordinates = {}
            focus_point_coordinate = multiply_matrices(
                transformation_matrix, air_matrix_coordinates["focus"]
            )
            for corner_name, corner_air_coordinate in air_matrix_coordinates.items():
                if corner_name == "focus":
                    continue

                corner_air_coordinate = multiply_matrices(
                    transformation_matrix, corner_air_coordinate
                )
                directing_vector = get_vector_by_2_points(
                    Point3d(
                        x=corner_air_coordinate[0],
                        y=corner_air_coordinate[1],
                        z=corner_air_coordinate[2],
                    ),
                    Point3d(
                        x=focus_point_coordinate[0],
                        y=focus_point_coordinate[1],
                        z=focus_point_coordinate[2],
                    ),
                )
                param = (
                    -1 * image.location.alt - focus_point_coordinate[2]
                ) / directing_vector.z
                ground_corner_coordinate = Point3d(
                    x=focus_point_coordinate[0] + directing_vector.x * param,
                    y=focus_point_coordinate[1] + directing_vector.y * param,
                    z=0,
                )
                ground_image_coordinates[cross_corner_mapping[corner_name]] = Point3d(
                    x=image.location.lot
                    + ground_corner_coordinate.x / LONGTITUDE_DISTANCE,
                    y=image.location.lat
                    + ground_corner_coordinate.y / LATITUDE_DISTANCE,
                    z=0,
                )

            ground_area = GroundArea(
                upper_right=ground_image_coordinates["upper_right"],
                upper_left=ground_image_coordinates["upper_left"],
                lower_right=ground_image_coordinates["lower_right"],
                lower_left=ground_image_coordinates["lower_left"],
            )
            image.ground_area = ground_area

        return images

    def _get_camera_matrix_coordinates_for_image(self, center: Point3d):
        upper_right_coordinate = np.array(
            [
                center.x + convert_mm_to_m(MATRIX_WIDTH) / 2,
                center.y + convert_mm_to_m(MATRIX_HEIGHT) / 2,
                center.z,
                1,
            ]
        )

        upper_left_coordinate = np.array(
            [
                center.x - convert_mm_to_m(MATRIX_WIDTH) / 2,
                center.y + convert_mm_to_m(MATRIX_HEIGHT) / 2,
                center.z,
                1,
            ]
        )

        lower_right_coordinate = np.array(
            [
                center.x + convert_mm_to_m(MATRIX_WIDTH) / 2,
                center.y - convert_mm_to_m(MATRIX_HEIGHT) / 2,
                center.z,
                1,
            ]
        )

        lower_left_coordinate = np.array(
            [
                center.x - convert_mm_to_m(MATRIX_WIDTH) / 2,
                center.y - convert_mm_to_m(MATRIX_HEIGHT) / 2,
                center.z,
                1,
            ]
        )

        focus_coordinate = np.array(
            [
                center.x,
                center.y,
                center.z - convert_mm_to_m(FOCAL_LENGTH),
                1,
            ]
        )

        return {
            "upper_left": upper_left_coordinate,
            "upper_right": upper_right_coordinate,
            "lower_left": lower_left_coordinate,
            "lower_right": lower_right_coordinate,
            "focus": focus_coordinate,
        }

    def visualize_images(self, output_file: str):
        print(f"Start generating {output_file}...")
        map_obj = folium.Map(
            location=[self.images[0].location.lat, self.images[0].location.lot],
            zoom_start=15,
        )

        for image in self.images:
            folium.Polygon(
                [
                    (image.ground_area.upper_right.y, image.ground_area.upper_right.x),
                    (image.ground_area.upper_left.y, image.ground_area.upper_left.x),
                    (image.ground_area.lower_left.y, image.ground_area.lower_left.x),
                    (image.ground_area.lower_right.y, image.ground_area.lower_right.x),
                ],
                color="blue",
                weight=2,
                fill=True,
                fill_color="orange",
                fill_opacity=0.4,
            ).add_to(map_obj)

        map_obj.save(output_file)
        print(f"Finish generating {output_file}!")

    def _generate_polygons_for_ground_area(self):
        polygons = []
        for image in self.images:
            curr_polygon = Polygon(
                Point(
                    image.ground_area.upper_right.x,
                    image.ground_area.upper_right.y,
                ),
                Point(
                    image.ground_area.upper_left.x,
                    image.ground_area.upper_left.y,
                ),
                Point(
                    image.ground_area.lower_left.x,
                    image.ground_area.lower_left.y,
                ),
                Point(
                    image.ground_area.lower_right.x,
                    image.ground_area.lower_right.y,
                ),
            )
            polygons.append(curr_polygon)

        return polygons
