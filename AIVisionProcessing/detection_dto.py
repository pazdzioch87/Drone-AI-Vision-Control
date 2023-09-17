from dataclasses import dataclass

@dataclass
class DetectionDto:
    confidence: float = None
    label: str = None
    coords: list = None
