from dataclasses import dataclass
from typing import Optional

from models.ground_area import GroundArea
from models.location import Location
from models.rotation import Rotation


@dataclass
class Image:
    id: int
    path: str
    location: Location
    rotation: Rotation
    ground_area: Optional[GroundArea] = None
