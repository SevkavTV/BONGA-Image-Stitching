from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    lat: float
    lot: float
    alt: float
