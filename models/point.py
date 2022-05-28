from dataclasses import dataclass


@dataclass
class Point2d:
    x: float
    y: float


@dataclass
class Point3d(Point2d):
    z: float
