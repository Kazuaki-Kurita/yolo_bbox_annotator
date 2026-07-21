from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from .models import Annotation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class AnnotationFormatError(ValueError):
    pass


def discover_images(images_dir: Path) -> list[Path]:
    """Return image paths relative to *images_dir* in stable natural order."""
    if not images_dir.is_dir():
        return []

    def natural_key(path: Path) -> list[tuple[int, object]]:
        import re

        parts = re.split(r"(\d+)", path.as_posix().lower())
        return [(1, int(part)) if part.isdigit() else (0, part) for part in parts]

    images = [
        path.relative_to(images_dir)
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_key)


def annotation_relative_path(image_relative_path: Path) -> Path:
    return image_relative_path.with_suffix(".txt")


def load_classes_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        name
        for name in (
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
        )
        if name
    ]


def _finite_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise AnnotationFormatError(f"{field} が数値ではありません: {value!r}") from exc
    if not math.isfinite(result):
        raise AnnotationFormatError(f"{field} は有限値である必要があります: {value!r}")
    return result


def _class_id(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise AnnotationFormatError(f"class_id が整数ではありません: {value!r}") from exc
    if result < 0:
        raise AnnotationFormatError(f"class_id は0以上である必要があります: {result}")
    return result


def _secondary_class_id(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise AnnotationFormatError(
            f"secondary_class_id が整数ではありません: {value!r}"
        ) from exc
    if result < -1:
        raise AnnotationFormatError(
            f"secondary_class_id は-1以上である必要があります: {result}"
        )
    return result


def _flag(value: str, field: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise AnnotationFormatError(f"{field} は0または1である必要があります: {value!r}")


def yolo_to_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if image_width <= 0 or image_height <= 0:
        raise AnnotationFormatError("画像サイズが不正です")
    if width <= 0 or height <= 0:
        raise AnnotationFormatError("BBoxのwidth/heightは0より大きい必要があります")
    if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
        raise AnnotationFormatError("YOLO座標は0〜1の範囲である必要があります")

    x_min = int(round((x_center - width / 2.0) * image_width))
    y_min = int(round((y_center - height / 2.0) * image_height))
    x_max = int(round((x_center + width / 2.0) * image_width))
    y_max = int(round((y_center + height / 2.0) * image_height))
    x_min = max(0, min(x_min, image_width))
    y_min = max(0, min(y_min, image_height))
    x_max = max(0, min(x_max, image_width))
    y_max = max(0, min(y_max, image_height))
    if x_max <= x_min or y_max <= y_min:
        raise AnnotationFormatError("BBoxが画像上で有効な面積を持ちません")
    return x_min, y_min, x_max, y_max


def bbox_to_yolo(
    annotation: Annotation, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    if not annotation.is_valid(image_width, image_height):
        raise AnnotationFormatError(f"画像範囲外または不正なBBoxです: {annotation.bbox}")
    width = annotation.x_max - annotation.x_min
    height = annotation.y_max - annotation.y_min
    return (
        (annotation.x_min + annotation.x_max) / 2.0 / image_width,
        (annotation.y_min + annotation.y_max) / 2.0 / image_height,
        width / image_width,
        height / image_height,
    )


def parse_annotation_line(
    line: str,
    image_width: int,
    image_height: int,
    *,
    extended: bool,
) -> Annotation:
    parts = line.strip().split(maxsplit=7)
    expected = 8 if extended else 5
    if len(parts) != expected:
        raise AnnotationFormatError(
            f"列数が不正です（期待: {expected}, 実際: {len(parts)}）"
        )
    class_id = _class_id(parts[0])
    x_center = _finite_float(parts[1], "x_center")
    y_center = _finite_float(parts[2], "y_center")
    width = _finite_float(parts[3], "width")
    height = _finite_float(parts[4], "height")
    bbox = yolo_to_bbox(x_center, y_center, width, height, image_width, image_height)

    secondary_class_id = -1
    certain = True
    visible = True
    if extended:
        try:
            secondary_class_id = _secondary_class_id(parts[5])
            certain = _flag(parts[6], "is_certain")
            visible = _flag(parts[7], "is_visible")
        except AnnotationFormatError as new_format_error:
            # Migration support for the previous format:
            # class bbox(4) certain visible note(JSON string)
            try:
                certain = _flag(parts[5], "certain")
                visible = _flag(parts[6], "visible")
                old_note = json.loads(parts[7])
                if not isinstance(old_note, str):
                    raise AnnotationFormatError(
                        "旧形式のnoteはJSON文字列である必要があります"
                    )
            except (AnnotationFormatError, json.JSONDecodeError):
                raise new_format_error
            secondary_class_id = -1

    if certain:
        secondary_class_id = class_id
    return Annotation(
        class_id,
        *bbox,
        secondary_class_id=secondary_class_id,
        certain=certain,
        visible=visible,
    )


def load_annotations(
    path: Path,
    image_width: int,
    image_height: int,
    *,
    extended: bool = False,
) -> tuple[list[Annotation], list[str]]:
    if not path.is_file():
        return [], []
    annotations: list[Annotation] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            annotations.append(
                parse_annotation_line(
                    raw_line, image_width, image_height, extended=extended
                )
            )
        except AnnotationFormatError as exc:
            errors.append(f"{path.name}:{line_number}: {exc}")
    return annotations, errors


def annotation_to_yolo_line(
    annotation: Annotation,
    image_width: int,
    image_height: int,
    *,
    extended: bool = False,
) -> str:
    x_center, y_center, width, height = bbox_to_yolo(
        annotation, image_width, image_height
    )
    base = (
        f"{annotation.class_id} {x_center:.8f} {y_center:.8f} "
        f"{width:.8f} {height:.8f}"
    )
    if not extended:
        return base
    secondary_class_id = (
        annotation.class_id if annotation.certain else annotation.secondary_class_id
    )
    return (
        f"{base} {secondary_class_id} "
        f"{int(annotation.certain)} {int(annotation.visible)}"
    )


def save_annotations(
    path: Path,
    annotations: Iterable[Annotation],
    image_width: int,
    image_height: int,
    *,
    extended: bool = False,
) -> None:
    lines = [
        annotation_to_yolo_line(
            annotation,
            image_width,
            image_height,
            extended=extended,
        )
        for annotation in annotations
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def save_classes(output_dir: Path, classes: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = [name.strip() for name in classes if name.strip()]
    output_dir.joinpath("classes.txt").write_text(
        "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
    )
