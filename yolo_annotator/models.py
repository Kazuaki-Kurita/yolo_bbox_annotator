from __future__ import annotations

from dataclasses import dataclass, replace


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class Annotation:
    """One bounding box and its review metadata in source-image pixels."""

    class_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    secondary_class_id: int = -1
    certain: bool = True
    visible: bool = True

    @property
    def bbox(self) -> BBox:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def with_bbox(self, bbox: BBox) -> "Annotation":
        return replace(
            self,
            x_min=bbox[0],
            y_min=bbox[1],
            x_max=bbox[2],
            y_max=bbox[3],
        )

    def is_valid(self, image_width: int, image_height: int) -> bool:
        return (
            self.class_id >= 0
            and self.secondary_class_id >= -1
            and 0 <= self.x_min < self.x_max <= image_width
            and 0 <= self.y_min < self.y_max <= image_height
        )
