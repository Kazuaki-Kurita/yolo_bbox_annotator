from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yolo_annotator.models import Annotation
from yolo_annotator.storage import (
    AnnotationFormatError,
    annotation_relative_path,
    annotation_to_yolo_line,
    discover_images,
    load_annotations,
    load_classes_file,
    parse_annotation_line,
    save_annotations,
)


class AnnotationStorageTests(unittest.TestCase):
    def test_classes_file_is_loaded_from_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "classes.txt"
            path.write_text("\ufeffbloom\n\nfaded\n", encoding="utf-8")
            self.assertEqual(load_classes_file(path), ["bloom", "faded"])

    def test_standard_yolo_round_trip(self) -> None:
        annotation = Annotation(
            2,
            100,
            200,
            500,
            800,
            secondary_class_id=1,
            certain=False,
            visible=False,
        )
        line = annotation_to_yolo_line(annotation, 3840, 2160)
        loaded = parse_annotation_line(line, 3840, 2160, extended=False)
        self.assertEqual(loaded.bbox, annotation.bbox)
        self.assertEqual(loaded.class_id, 2)
        self.assertEqual(loaded.secondary_class_id, 2)
        self.assertTrue(loaded.certain)
        self.assertTrue(loaded.visible)

    def test_extended_yolo_round_trip_preserves_metadata(self) -> None:
        annotation = Annotation(
            1,
            50,
            75,
            125,
            175,
            secondary_class_id=3,
            certain=False,
            visible=True,
        )
        line = annotation_to_yolo_line(annotation, 200, 250, extended=True)
        loaded = parse_annotation_line(line, 200, 250, extended=True)
        self.assertEqual(loaded, annotation)
        self.assertEqual(len(line.split()), 8)
        self.assertEqual(line.split()[-3:], ["3", "0", "1"])

    def test_certain_annotation_writes_primary_as_second_class(self) -> None:
        annotation = Annotation(
            1,
            50,
            75,
            125,
            175,
            secondary_class_id=3,
            certain=True,
            visible=False,
        )
        line = annotation_to_yolo_line(annotation, 200, 250, extended=True)
        self.assertEqual(line.split()[-3:], ["1", "1", "0"])

    def test_previous_extended_format_is_migrated(self) -> None:
        loaded = parse_annotation_line(
            '0 0.5 0.5 0.2 0.2 0 1 "旧note"',
            100,
            100,
            extended=True,
        )
        self.assertEqual(loaded.secondary_class_id, -1)
        self.assertFalse(loaded.certain)
        self.assertTrue(loaded.visible)

    def test_save_creates_empty_label_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "image.txt"
            save_annotations(path, [], 100, 100)
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_load_skips_bad_rows_and_reports_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.txt"
            path.write_text(
                "0 0.5 0.5 0.2 0.2\n"
                "bad row\n"
                "1 0.25 0.25 0.1 0.1\n",
                encoding="utf-8",
            )
            annotations, errors = load_annotations(path, 100, 100)
            self.assertEqual(len(annotations), 2)
            self.assertEqual(len(errors), 1)
            self.assertIn("image.txt:2", errors[0])

    def test_invalid_boolean_flag_is_rejected(self) -> None:
        with self.assertRaises(AnnotationFormatError):
            parse_annotation_line(
                "0 0.5 0.5 0.2 0.2 2 yes 1",
                100,
                100,
                extended=True,
            )

    def test_nested_images_keep_relative_label_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sub").mkdir()
            (root / "10.jpg").touch()
            (root / "sub" / "2.png").touch()
            (root / "ignore.txt").touch()
            images = discover_images(root)
            self.assertEqual(set(images), {Path("10.jpg"), Path("sub/2.png")})
            self.assertEqual(
                annotation_relative_path(Path("sub/2.png")), Path("sub/2.txt")
            )


if __name__ == "__main__":
    unittest.main()
