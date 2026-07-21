from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImageReader,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .models import Annotation, BBox


@dataclass(slots=True)
class DisplayTransform:
    target: QRectF
    source_width: int
    source_height: int

    def widget_to_source(self, point: QPoint) -> tuple[int, int] | None:
        if not self.target.contains(point):
            return None
        return self.widget_to_source_clamped(point)

    def widget_to_source_clamped(self, point: QPoint) -> tuple[int, int]:
        widget_x = min(max(float(point.x()), self.target.left()), self.target.right())
        widget_y = min(max(float(point.y()), self.target.top()), self.target.bottom())
        x_ratio = (widget_x - self.target.left()) / self.target.width()
        y_ratio = (widget_y - self.target.top()) / self.target.height()
        x = int(round(x_ratio * self.source_width))
        y = int(round(y_ratio * self.source_height))
        return (
            max(0, min(x, self.source_width)),
            max(0, min(y, self.source_height)),
        )

    def source_rect_to_widget(self, bbox: BBox) -> QRectF:
        x_min, y_min, x_max, y_max = bbox
        scale_x = self.target.width() / self.source_width
        scale_y = self.target.height() / self.source_height
        return QRectF(
            self.target.left() + x_min * scale_x,
            self.target.top() + y_min * scale_y,
            max(1.0, (x_max - x_min) * scale_x),
            max(1.0, (y_max - y_min) * scale_y),
        )


class AnnotationCanvas(QWidget):
    selection_changed = Signal(int)
    bbox_created = Signal(object)
    bbox_changed = Signal(int, object)

    _MIN_BBOX_SIZE = 2
    _HANDLE_HALF_SIZE = 4.0
    _HANDLE_HIT_MARGIN = 7.0
    _MIN_ZOOM = 1.0
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.25
    _COLORS = (
        QColor(0, 255, 110),
        QColor(255, 190, 0),
        QColor(80, 180, 255),
        QColor(210, 90, 255),
        QColor(255, 110, 80),
        QColor(80, 230, 230),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pixmap: QPixmap | None = None
        self._source_width = 0
        self._source_height = 0
        self._annotations: list[Annotation] = []
        self._class_names: list[str] = []
        self._selected_index = -1
        self._transform: DisplayTransform | None = None
        self._message = "imagesフォルダを読み込んでください"
        self._new_box_mode = False
        self._zoom_factor = self._MIN_ZOOM
        self._view_center = (0.0, 0.0)

        self._drag_mode: str | None = None
        self._resize_handle: str | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self._bbox_before_drag: BBox | None = None
        self._press_widget_position: QPointF | None = None
        self._drag_has_moved = False

    @property
    def image_size(self) -> tuple[int, int]:
        return self._source_width, self._source_height

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def set_message(self, message: str) -> None:
        self._message = message
        self.update()

    def set_image(self, path: Path) -> bool:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            self._pixmap = None
            self._source_width = 0
            self._source_height = 0
            self.set_message(f"画像を開けません: {reader.errorString()}")
            return False
        self._pixmap = QPixmap.fromImage(image)
        self._source_width = image.width()
        self._source_height = image.height()
        self._new_box_mode = False
        self._zoom_factor = self._MIN_ZOOM
        self._view_center = (
            self._source_width / 2.0,
            self._source_height / 2.0,
        )
        self._reset_drag_state()
        self.update()
        return True

    def set_annotations(
        self,
        annotations: list[Annotation],
        class_names: list[str],
        selected_index: int = -1,
    ) -> None:
        self._annotations = annotations
        self._class_names = class_names
        self._selected_index = (
            selected_index if 0 <= selected_index < len(annotations) else -1
        )
        self._new_box_mode = False
        self._reset_drag_state()
        self.update()

    def set_selected_index(self, index: int, *, emit: bool = False) -> None:
        normalized = index if 0 <= index < len(self._annotations) else -1
        if normalized == self._selected_index:
            return
        self._selected_index = normalized
        self.update()
        if emit:
            self.selection_changed.emit(normalized)

    def begin_new_bbox(self) -> None:
        self._new_box_mode = True
        self._reset_drag_state(keep_new_mode=True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def cancel_new_bbox(self) -> None:
        self._new_box_mode = False
        self._reset_drag_state()
        self.update()

    def reset_zoom(self) -> None:
        """Return to the full-image view."""
        self._zoom_factor = self._MIN_ZOOM
        self._view_center = (
            self._source_width / 2.0,
            self._source_height / 2.0,
        )
        self.update()

    def _reset_drag_state(self, *, keep_new_mode: bool = False) -> None:
        self._drag_mode = None
        self._resize_handle = None
        self._drag_start = None
        self._drag_current = None
        self._bbox_before_drag = None
        self._press_widget_position = None
        self._drag_has_moved = False
        if not keep_new_mode:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _compute_transform(self) -> DisplayTransform | None:
        if self._pixmap is None or self._source_width <= 0 or self._source_height <= 0:
            return None
        area = self.rect()
        fit_scale = min(
            area.width() / self._source_width,
            area.height() / self._source_height,
        )
        scale = fit_scale * self._zoom_factor
        center_x, center_y = self._clamp_view_center(
            self._view_center[0], self._view_center[1], scale
        )
        self._view_center = center_x, center_y
        width = self._source_width * scale
        height = self._source_height * scale
        return DisplayTransform(
            QRectF(
                area.center().x() - center_x * scale,
                area.center().y() - center_y * scale,
                width,
                height,
            ),
            self._source_width,
            self._source_height,
        )

    def _clamp_view_center(
        self, center_x: float, center_y: float, scale: float
    ) -> tuple[float, float]:
        """Keep large images covering, and small images inside, the viewport."""
        if scale <= 0.0:
            return self._source_width / 2.0, self._source_height / 2.0

        half_visible_width = self.width() / scale / 2.0
        half_visible_height = self.height() / scale / 2.0
        x_edge = self._source_width - half_visible_width
        y_edge = self._source_height - half_visible_height
        x_min, x_max = sorted((half_visible_width, x_edge))
        y_min, y_max = sorted((half_visible_height, y_edge))
        center_x = min(max(center_x, x_min), x_max)
        center_y = min(max(center_y, y_min), y_max)
        return center_x, center_y

    def zoom_at(self, position: QPointF, wheel_delta: int) -> None:
        """Zoom around a widget position while keeping its source point fixed."""
        if (
            wheel_delta == 0
            or self._pixmap is None
            or self._source_width <= 0
            or self._source_height <= 0
        ):
            return
        transform = self._compute_transform()
        if transform is None:
            return

        widget_x = min(max(position.x(), transform.target.left()), transform.target.right())
        widget_y = min(max(position.y(), transform.target.top()), transform.target.bottom())
        source_x = (
            (widget_x - transform.target.left())
            / transform.target.width()
            * self._source_width
        )
        source_y = (
            (widget_y - transform.target.top())
            / transform.target.height()
            * self._source_height
        )

        new_zoom = self._zoom_factor * (
            self._ZOOM_STEP ** (wheel_delta / 120.0)
        )
        new_zoom = min(max(new_zoom, self._MIN_ZOOM), self._MAX_ZOOM)
        if abs(new_zoom - self._zoom_factor) < 1e-9:
            return
        self._zoom_factor = new_zoom

        if new_zoom <= self._MIN_ZOOM:
            self._view_center = (
                self._source_width / 2.0,
                self._source_height / 2.0,
            )
        else:
            area = self.rect()
            fit_scale = min(
                area.width() / self._source_width,
                area.height() / self._source_height,
            )
            new_scale = fit_scale * new_zoom
            desired_center_x = source_x + (area.center().x() - position.x()) / new_scale
            desired_center_y = source_y + (area.center().y() - position.y()) / new_scale
            self._view_center = self._clamp_view_center(
                desired_center_x, desired_center_y, new_scale
            )
        self.update()

    @staticmethod
    def _normalized_bbox(first: tuple[int, int], second: tuple[int, int]) -> BBox:
        x1, y1 = first
        x2, y2 = second
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    @classmethod
    def _valid_bbox(cls, bbox: BBox | None) -> bool:
        return bool(
            bbox
            and bbox[2] - bbox[0] >= cls._MIN_BBOX_SIZE
            and bbox[3] - bbox[1] >= cls._MIN_BBOX_SIZE
        )

    def _corner_points(self, bbox: BBox) -> dict[str, QPointF]:
        if self._transform is None:
            return {}
        rect = self._transform.source_rect_to_widget(bbox)
        return {
            "top_left": rect.topLeft(),
            "top_right": rect.topRight(),
            "bottom_right": rect.bottomRight(),
            "bottom_left": rect.bottomLeft(),
        }

    def _hit_test_handle(self, point: QPointF) -> str | None:
        if not 0 <= self._selected_index < len(self._annotations):
            return None
        radius = self._HANDLE_HALF_SIZE + self._HANDLE_HIT_MARGIN
        bbox = self._annotations[self._selected_index].bbox
        for name, corner in self._corner_points(bbox).items():
            if abs(point.x() - corner.x()) <= radius and abs(point.y() - corner.y()) <= radius:
                return name
        return None

    def _hit_test_bbox(self, source: tuple[int, int]) -> int:
        x, y = source
        order = list(range(len(self._annotations) - 1, -1, -1))
        if self._selected_index in order:
            order.remove(self._selected_index)
            order.insert(0, self._selected_index)
        for index in order:
            x_min, y_min, x_max, y_max = self._annotations[index].bbox
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return index
        return -1

    @staticmethod
    def _opposite_corner(bbox: BBox, handle: str) -> tuple[int, int]:
        x_min, y_min, x_max, y_max = bbox
        return {
            "top_left": (x_max, y_max),
            "top_right": (x_min, y_max),
            "bottom_right": (x_min, y_min),
            "bottom_left": (x_max, y_min),
        }[handle]

    @staticmethod
    def _cursor_for_handle(handle: str | None) -> Qt.CursorShape:
        if handle in {"top_left", "bottom_right"}:
            return Qt.CursorShape.SizeFDiagCursor
        if handle in {"top_right", "bottom_left"}:
            return Qt.CursorShape.SizeBDiagCursor
        return Qt.CursorShape.ArrowCursor

    def _preview_bbox(self) -> BBox | None:
        if self._drag_start is None or self._drag_current is None:
            return None
        if self._drag_mode in {"draw", "resize"}:
            return self._normalized_bbox(self._drag_start, self._drag_current)
        if self._drag_mode == "move" and self._bbox_before_drag is not None:
            dx = self._drag_current[0] - self._drag_start[0]
            dy = self._drag_current[1] - self._drag_start[1]
            x_min, y_min, x_max, y_max = self._bbox_before_drag
            width = x_max - x_min
            height = y_max - y_min
            new_x_min = max(0, min(x_min + dx, self._source_width - width))
            new_y_min = max(0, min(y_min + dy, self._source_height - height))
            return new_x_min, new_y_min, new_x_min + width, new_y_min + height
        return None

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        self._transform = self._compute_transform()
        if self._pixmap is None or self._transform is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        painter.drawPixmap(self._transform.target, self._pixmap, QRectF(self._pixmap.rect()))
        preview = self._preview_bbox()
        for index, annotation in enumerate(self._annotations):
            bbox = annotation.bbox
            if index == self._selected_index and preview is not None and self._drag_mode in {"move", "resize"}:
                bbox = preview
            rect = self._transform.source_rect_to_widget(bbox)
            selected = index == self._selected_index
            color = QColor(255, 40, 40) if selected else self._COLORS[annotation.class_id % len(self._COLORS)]
            style = Qt.PenStyle.SolidLine if annotation.visible else Qt.PenStyle.DashLine
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.setPen(QPen(color, 3 if selected else 2, style))
            painter.drawRect(rect)

            class_name = (
                self._class_names[annotation.class_id]
                if 0 <= annotation.class_id < len(self._class_names)
                else f"class_{annotation.class_id}"
            )
            secondary_name = (
                self._class_names[annotation.secondary_class_id]
                if 0 <= annotation.secondary_class_id < len(self._class_names)
                else None
            )
            flags = []
            if not annotation.certain:
                flags.append("uncertain")
            if not annotation.visible:
                flags.append("not visible")
            label = f"{index + 1}: {class_name}"
            if not annotation.certain and secondary_name is not None:
                label += f" / {secondary_name}"
            if flags:
                label += " [" + ", ".join(flags) + "]"
            metrics = painter.fontMetrics()
            text_rect = QRectF(metrics.boundingRect(label))
            text_rect.adjust(-4.0, -2.0, 4.0, 2.0)
            text_rect.moveBottomLeft(
                QPointF(rect.left(), max(self._transform.target.top() + text_rect.height(), rect.top()))
            )
            painter.setPen(QPen(QColor(0, 0, 0, 0), 0))
            painter.setBrush(QBrush(QColor(0, 0, 0, 185)))
            painter.drawRect(text_rect)
            painter.setPen(QPen(color, 1))
            painter.drawText(
                text_rect.adjusted(4.0, 2.0, -4.0, -2.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            if selected:
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.setBrush(QBrush(color))
                for corner in self._corner_points(bbox).values():
                    painter.drawRect(
                        QRectF(
                            corner.x() - self._HANDLE_HALF_SIZE,
                            corner.y() - self._HANDLE_HALF_SIZE,
                            self._HANDLE_HALF_SIZE * 2.0,
                            self._HANDLE_HALF_SIZE * 2.0,
                        )
                    )

        if self._drag_mode == "draw" and self._valid_bbox(preview):
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.setPen(QPen(QColor(255, 40, 40), 3))
            painter.drawRect(self._transform.source_rect_to_widget(preview))

        if self._new_box_mode and self._drag_mode is None:
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(
                self._transform.target.adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                "新規BBox: 画像上をドラッグ（Escで解除）",
            )

        zoom_text = f"Zoom {self._zoom_factor * 100:.0f}%  (Ctrl+ホイール)"
        metrics = painter.fontMetrics()
        zoom_rect = QRectF(
            self.width() - metrics.horizontalAdvance(zoom_text) - 22.0,
            10.0,
            metrics.horizontalAdvance(zoom_text) + 12.0,
            metrics.height() + 8.0,
        )
        painter.setPen(QPen(QColor(0, 0, 0, 0), 0))
        painter.setBrush(QBrush(QColor(0, 0, 0, 185)))
        painter.drawRect(zoom_rect)
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawText(
            zoom_rect,
            Qt.AlignmentFlag.AlignCenter,
            zoom_text,
        )

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_at(event.position(), event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._transform is None:
            return
        source = self._transform.widget_to_source(event.position().toPoint())
        if source is None:
            return
        self.setFocus()
        self._press_widget_position = event.position()
        self._drag_has_moved = False

        if self._new_box_mode:
            self._drag_mode = "draw"
            self._drag_start = source
            self._drag_current = source
            return

        handle = self._hit_test_handle(event.position())
        if handle is not None and 0 <= self._selected_index < len(self._annotations):
            self._drag_mode = "resize"
            self._resize_handle = handle
            self._bbox_before_drag = self._annotations[self._selected_index].bbox
            self._drag_start = self._opposite_corner(self._bbox_before_drag, handle)
            self._drag_current = source
            return

        hit_index = self._hit_test_bbox(source)
        if hit_index >= 0:
            if hit_index != self._selected_index:
                self._selected_index = hit_index
                self.selection_changed.emit(hit_index)
            self._drag_mode = "move"
            self._drag_start = source
            self._drag_current = source
            self._bbox_before_drag = self._annotations[hit_index].bbox
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            if self._selected_index != -1:
                self._selected_index = -1
                self.selection_changed.emit(-1)
            self._drag_mode = "draw"
            self._drag_start = source
            self._drag_current = source
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._transform is None:
            return
        if self._drag_start is not None:
            if self._press_widget_position is not None:
                delta = event.position() - self._press_widget_position
                self._drag_has_moved = abs(delta.x()) >= 2.0 or abs(delta.y()) >= 2.0
            self._drag_current = self._transform.widget_to_source_clamped(
                event.position().toPoint()
            )
            self.update()
            return
        if self._new_box_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(self._cursor_for_handle(self._hit_test_handle(event.position())))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return
        if self._transform is not None:
            self._drag_current = self._transform.widget_to_source_clamped(
                event.position().toPoint()
            )
        preview = self._preview_bbox()
        mode = self._drag_mode
        selected = self._selected_index
        moved = self._drag_has_moved
        valid = self._valid_bbox(preview)
        self._new_box_mode = False
        self._reset_drag_state()
        if moved and valid and preview is not None:
            if mode == "draw":
                self.bbox_created.emit(preview)
            elif mode in {"move", "resize"} and selected >= 0:
                self.bbox_changed.emit(selected, preview)
        self.update()
