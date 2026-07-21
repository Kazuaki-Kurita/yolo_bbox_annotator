from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QLibraryInfo
from PySide6.QtWidgets import QApplication


def _configure_qt_plugin_path() -> None:
    """Use the platform plugins bundled with the active PySide6 install."""
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = os.environ.get(key, "")
        if "cv2" in value.replace("\\", "/").lower():
            os.environ.pop(key, None)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.path(
        QLibraryInfo.LibraryPath.PluginsPath
    )


def _workspace_default() -> Path | None:
    dataset = Path(__file__).resolve().parent.parent / "datasets_refresh300"
    if (
        (dataset / "images").is_dir()
        and (dataset / "labels").is_dir()
        and (dataset / "classes.txt").is_file()
    ):
        return dataset
    return None


def parse_args() -> argparse.Namespace:
    default_dataset = _workspace_default()
    parser = argparse.ArgumentParser(description="YOLO image bounding-box annotator")
    parser.add_argument(
        "--dataset-dir",
        default=str(default_dataset) if default_dataset else None,
        help="Dataset root containing images, labels, and classes.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_qt_plugin_path()
    from yolo_annotator.main_window import MainWindow

    _configure_qt_plugin_path()
    app = QApplication(sys.argv)
    window = MainWindow(args.dataset_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
