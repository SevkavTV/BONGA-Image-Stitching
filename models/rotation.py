from dataclasses import dataclass


@dataclass(frozen=True)
class Rotation:
    yaw: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
