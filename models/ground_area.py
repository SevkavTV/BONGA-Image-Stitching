from dataclasses import dataclass

from models.geo import Point3d


@dataclass
class GroundArea:
    upper_right: Point3d
    upper_left: Point3d
    lower_right: Point3d
    lower_left: Point3d
