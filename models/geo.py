from dataclasses import dataclass


@dataclass
class Point2d:
    x: float
    y: float


@dataclass
class Point3d(Point2d):
    z: float


@dataclass
class Vector(Point3d):
    ...


@dataclass
class Plane:
    a: float
    b: float
    c: float
    d: float
