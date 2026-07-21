from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction, QCloseEvent, QImageReader, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .models import Annotation, BBox
from .storage import (
    annotation_relative_path,
    discover_images,
    load_annotations,
    load_classes_file,
    save_annotations,
    save_classes,
)
from .widgets import AnnotationCanvas


class MainWindow(QMainWindow):
    def __init__(
        self,
        dataset_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("YOLO BBox Annotator")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.dataset_dir = Path(dataset_dir).resolve() if dataset_dir else None
        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.classes_path: Path | None = None
        self.output_dir: Path | None = None
        self._sync_dataset_paths()
        self.images: list[Path] = []
        self.classes: list[str] = []
        self.annotations: dict[Path, list[Annotation]] = {}
        self.image_sizes: dict[Path, tuple[int, int]] = {}
        self.reviewed: set[Path] = set()
        self.dirty: set[Path] = set()
        self.load_errors: list[str] = []
        self.current_image_index = -1
        self.selected_annotation_index = -1
        self._loading_form = False
        self._loading_image = False

        self._build_ui()
        self._build_menu_and_shortcuts()
        self._apply_initial_window_size()
        self._update_path_labels()

        if self.dataset_dir:
            self.load_dataset()

    # ---------- UI ----------

    def _apply_initial_window_size(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 850)
            return
        available = screen.availableGeometry()
        self.resize(
            min(1600, max(1000, int(available.width() * 0.94))),
            min(950, max(650, int(available.height() * 0.94))),
        )

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        input_bar = QHBoxLayout()
        dataset_button = QPushButton("入力フォルダを選択")
        dataset_button.clicked.connect(self.choose_dataset_dir)
        load_button = QPushButton("データセットを読み込み")
        load_button.clicked.connect(self.load_dataset)
        save_button = QPushButton("全画像を保存 [Ctrl+S]")
        save_button.clicked.connect(self.save_all)
        for button in (dataset_button, load_button, save_button):
            input_bar.addWidget(button)
        root.addLayout(input_bar)

        path_bar = QHBoxLayout()
        self.dataset_path_label = QLabel()
        self.output_path_label = QLabel()
        for label in (self.dataset_path_label, self.output_path_label):
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            path_bar.addWidget(label, 1)
        root.addLayout(path_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1000, 600])
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("入力フォルダを選択してください")

    def _build_image_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnnotationCanvas()
        self.canvas.selection_changed.connect(self._canvas_selection_changed)
        self.canvas.bbox_created.connect(self._bbox_created)
        self.canvas.bbox_changed.connect(self._bbox_changed)
        layout.addWidget(self.canvas, 1)

        nav = QHBoxLayout()
        previous_button = QPushButton("← 前の画像 [Ctrl+←]")
        previous_button.clicked.connect(lambda: self.navigate(-1))
        next_button = QPushButton("次の画像 [Ctrl+→] →")
        next_button.clicked.connect(lambda: self.navigate(1))
        self.image_spin = QSpinBox()
        self.image_spin.setRange(0, 0)
        self.image_spin.setSpecialValueText("-")
        self.image_spin.valueChanged.connect(self._image_spin_changed)
        self.image_count_label = QLabel("/ 0")
        reset_zoom_button = QPushButton("全体表示 [Ctrl+0]")
        reset_zoom_button.clicked.connect(self.canvas.reset_zoom)
        new_button = QPushButton("新規BBox [Ctrl+N]")
        new_button.clicked.connect(self.begin_new_bbox)
        delete_button = QPushButton("選択BBoxを削除 [Ctrl+D]")
        delete_button.clicked.connect(self.delete_selected_bbox)
        nav.addWidget(previous_button)
        nav.addWidget(self.image_spin)
        nav.addWidget(self.image_count_label)
        nav.addWidget(next_button)
        nav.addWidget(reset_zoom_button)
        nav.addStretch(1)
        nav.addWidget(new_button)
        nav.addWidget(delete_button)
        layout.addLayout(nav)

        info = QHBoxLayout()
        self.image_name_label = QLabel("画像: -")
        self.image_resolution_label = QLabel("解像度: -")
        self.current_bbox_label = QLabel("BBox: -")
        info.addWidget(self.image_name_label, 2)
        info.addWidget(self.image_resolution_label)
        info.addWidget(self.current_bbox_label, 2)
        layout.addLayout(info)
        return panel

    def _build_right_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_annotation_tab(), "アノテーション")
        tabs.addTab(self._build_dataset_tab(), "画像一覧・読込情報")
        return tabs

    def _build_annotation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("選択中BBoxの情報（変更は自動反映）")
        form = QFormLayout(group)
        self.label_combo = QComboBox()
        self.label_combo.currentIndexChanged.connect(self._form_changed)
        form.addRow("label", self.label_combo)

        self.secondary_label_combo = QComboBox()
        self.secondary_label_combo.setEnabled(False)
        self.secondary_label_combo.currentIndexChanged.connect(self._form_changed)
        form.addRow("第2 label", self.secondary_label_combo)

        flags = QWidget()
        flags_layout = QHBoxLayout(flags)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        self.certain_check = QCheckBox("certain")
        self.visible_check = QCheckBox("visible")
        self.certain_check.setChecked(True)
        self.visible_check.setChecked(True)
        self.certain_check.toggled.connect(self._certain_toggled)
        self.visible_check.toggled.connect(self._form_changed)
        flags_layout.addWidget(self.certain_check)
        flags_layout.addWidget(self.visible_check)
        flags_layout.addStretch(1)
        form.addRow("flags", flags)

        layout.addWidget(group)

        bbox_actions = QHBoxLayout()
        new_button = QPushButton("新規BBox [Ctrl+N]")
        new_button.clicked.connect(self.begin_new_bbox)
        delete_button = QPushButton("選択BBoxを削除 [Ctrl+D]")
        delete_button.clicked.connect(self.delete_selected_bbox)
        bbox_actions.addWidget(new_button)
        bbox_actions.addWidget(delete_button)
        layout.addLayout(bbox_actions)

        self.bbox_table = QTableWidget(0, 6)
        self.bbox_table.setHorizontalHeaderLabels(
            ["#", "label", "第2 label", "certain", "visible", "bbox(px)"]
        )
        self.bbox_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.bbox_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.bbox_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.bbox_table.verticalHeader().setVisible(False)
        self.bbox_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.bbox_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.bbox_table.itemSelectionChanged.connect(self._bbox_table_selection_changed)
        layout.addWidget(QLabel("現在画像のBBox一覧"))
        layout.addWidget(self.bbox_table, 1)
        return tab

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.dataset_summary_label = QLabel("未読込")
        layout.addWidget(self.dataset_summary_label)
        self.image_table = QTableWidget(0, 4)
        self.image_table.setHorizontalHeaderLabels(["#", "画像", "BBox数", "出力済み"])
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.image_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.image_table.verticalHeader().setVisible(False)
        self.image_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.image_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.image_table.cellDoubleClicked.connect(self._image_table_activated)
        layout.addWidget(self.image_table, 2)
        layout.addWidget(QLabel("読込時の警告（不正行はスキップ）"))
        self.error_text = QPlainTextEdit()
        self.error_text.setReadOnly(True)
        self.error_text.setMaximumBlockCount(1000)
        layout.addWidget(self.error_text, 1)
        return tab

    def _build_menu_and_shortcuts(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル")
        open_action = QAction("データセットを読込", self)
        open_action.triggered.connect(self.load_dataset)
        save_action = QAction("全画像を保存", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_all)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        shortcuts = (
            ("Ctrl+Left", lambda: self.navigate(-1)),
            ("Ctrl+Right", lambda: self.navigate(1)),
            ("Ctrl+N", self.begin_new_bbox),
            ("Ctrl+D", self.delete_selected_bbox),
            ("Ctrl+0", self.canvas.reset_zoom),
            ("Escape", self.canvas.cancel_new_bbox),
        )
        self._shortcuts: list[QShortcut] = []
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    # ---------- directory and dataset loading ----------

    def _choose_dir(self, title: str, initial: Path | None) -> Path | None:
        selected = QFileDialog.getExistingDirectory(
            self,
            title,
            str(initial or Path.cwd()),
            QFileDialog.Option.ShowDirsOnly,
        )
        return Path(selected).resolve() if selected else None

    def _sync_dataset_paths(self) -> None:
        if self.dataset_dir is None:
            self.images_dir = None
            self.labels_dir = None
            self.classes_path = None
            self.output_dir = None
            return
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        self.classes_path = self.dataset_dir / "classes.txt"
        self.output_dir = self.dataset_dir / "annotation_output"

    def choose_dataset_dir(self) -> None:
        selected = self._choose_dir("入力フォルダを選択", self.dataset_dir)
        if selected is None or not self._confirm_discard_dirty():
            return
        self.dataset_dir = selected
        self._sync_dataset_paths()
        self._update_path_labels()

    def _update_path_labels(self) -> None:
        entries = (
            (self.dataset_path_label, "入力", self.dataset_dir),
            (self.output_path_label, "出力", self.output_dir),
        )
        for label, prefix, path in entries:
            text = str(path) if path else "未選択"
            label.setText(f"{prefix}: {text}")
            label.setToolTip(text)

    def _confirm_discard_dirty(self) -> bool:
        if not self.dirty:
            return True
        response = QMessageBox.question(
            self,
            "未保存の変更",
            "未保存の変更があります。現在の出力先へ保存してから再読込しますか？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if response == QMessageBox.StandardButton.Cancel:
            return False
        if response == QMessageBox.StandardButton.Save:
            return self.save_all(show_dialog=False)
        self.dirty.clear()
        return True

    def load_dataset(self) -> None:
        if not self._confirm_discard_dirty():
            return
        if self.dataset_dir is None or not self.dataset_dir.is_dir():
            QMessageBox.warning(self, "入力エラー", "有効な入力フォルダを選択してください。")
            return
        self._sync_dataset_paths()
        assert self.images_dir is not None
        assert self.labels_dir is not None
        assert self.classes_path is not None
        assert self.output_dir is not None

        missing: list[str] = []
        if not self.images_dir.is_dir():
            missing.append("images/")
        if not self.labels_dir.is_dir():
            missing.append("labels/")
        if not self.classes_path.is_file():
            missing.append("classes.txt")
        if missing:
            QMessageBox.warning(
                self,
                "入力エラー",
                "入力フォルダ内に必要な項目がありません: " + ", ".join(missing),
            )
            return

        images = discover_images(self.images_dir)
        if not images:
            QMessageBox.warning(self, "入力エラー", "imagesフォルダ内に画像がありません。")
            return

        self.images = images
        self.classes = load_classes_file(self.classes_path)
        if not self.classes:
            QMessageBox.warning(
                self,
                "入力エラー",
                "classes.txtにクラス名がありません。",
            )
            return
        self.annotations.clear()
        self.image_sizes.clear()
        self.reviewed.clear()
        self.dirty.clear()
        self.load_errors.clear()

        output_labels = self.output_dir / "labels"
        output_extended = self.output_dir / "labels_extended"
        max_class_id = len(self.classes) - 1
        unreadable_images: list[Path] = []

        for relative in self.images:
            reader = QImageReader(str(self.images_dir / relative))
            size = reader.size()
            if not size.isValid():
                self.annotations[relative] = []
                unreadable_images.append(relative)
                self.load_errors.append(f"{relative}: 画像サイズを取得できません")
                continue
            width, height = size.width(), size.height()
            self.image_sizes[relative] = (width, height)
            label_relative = annotation_relative_path(relative)
            extended_path = output_extended / label_relative
            saved_path = output_labels / label_relative
            source_path = self.labels_dir / label_relative
            if extended_path.exists():
                selected_path, extended = extended_path, True
                self.reviewed.add(relative)
            elif saved_path.exists():
                selected_path, extended = saved_path, False
                self.reviewed.add(relative)
            else:
                selected_path, extended = source_path, False
            loaded, errors = load_annotations(
                selected_path, width, height, extended=extended
            )
            self.annotations[relative] = loaded
            self.load_errors.extend(errors)
            if loaded:
                max_class_id = max(
                    max_class_id,
                    max(
                        max(item.class_id, item.secondary_class_id)
                        for item in loaded
                    ),
                )

        while len(self.classes) <= max_class_id:
            self.classes.append(f"class_{len(self.classes)}")

        self._populate_label_combo()
        self._populate_image_table()
        self.error_text.setPlainText("\n".join(self.load_errors) or "警告はありません。")
        total_boxes = sum(len(items) for items in self.annotations.values())
        self.dataset_summary_label.setText(
            f"画像 {len(self.images)}枚 / BBox {total_boxes}個 / "
            f"出力再開 {len(self.reviewed)}枚 / 警告 {len(self.load_errors)}件"
        )
        self.image_spin.setRange(1, len(self.images))
        self.image_count_label.setText(f"/ {len(self.images)}")
        self.current_image_index = -1
        self._update_path_labels()
        self._show_image(0)
        self.statusBar().showMessage(
            f"{len(self.images)}枚・BBox {total_boxes}個を読み込みました"
        )
        if unreadable_images:
            QMessageBox.warning(
                self,
                "画像読込警告",
                f"{len(unreadable_images)}枚の画像サイズを取得できません。詳細は読込情報タブを確認してください。",
            )

    def _populate_label_combo(self) -> None:
        primary_blocker = QSignalBlocker(self.label_combo)
        secondary_blocker = QSignalBlocker(self.secondary_label_combo)
        self.label_combo.clear()
        self.secondary_label_combo.clear()
        self.secondary_label_combo.addItem("-1: なし", -1)
        for class_id, name in enumerate(self.classes):
            label = f"{class_id}: {name}"
            self.label_combo.addItem(label, class_id)
            self.secondary_label_combo.addItem(label, class_id)
        if self.certain_check.isChecked():
            primary_index = self.secondary_label_combo.findData(
                self.label_combo.currentData()
            )
            if primary_index >= 0:
                self.secondary_label_combo.setCurrentIndex(primary_index)
        del primary_blocker
        del secondary_blocker

    # ---------- navigation ----------

    def _image_spin_changed(self, value: int) -> None:
        if self._loading_image or not self.images:
            return
        self._show_image(value - 1)

    def navigate(self, delta: int) -> None:
        if not self.images:
            return
        target = max(0, min(self.current_image_index + delta, len(self.images) - 1))
        self._show_image(target)

    def _show_image(self, index: int) -> None:
        if not 0 <= index < len(self.images):
            return
        if self.current_image_index >= 0:
            current = self.images[self.current_image_index]
            if current in self.dirty and not self._save_one(current):
                return

        relative = self.images[index]
        self._loading_image = True
        self.current_image_index = index
        self.selected_annotation_index = -1
        self.canvas.cancel_new_bbox()
        loaded = self.canvas.set_image(self.images_dir / relative) if self.images_dir else False
        if loaded:
            actual_size = self.canvas.image_size
            expected_size = self.image_sizes.get(relative)
            if expected_size and expected_size != actual_size:
                self.load_errors.append(
                    f"{relative}: EXIF変換後サイズ {actual_size} がラベル読込サイズ {expected_size} と異なります"
                )
            self.image_sizes[relative] = actual_size
        self.canvas.set_annotations(self.annotations.get(relative, []), self.classes, -1)
        self._refresh_bbox_table()
        self._load_annotation_form(-1)
        with QSignalBlocker(self.image_spin):
            self.image_spin.setValue(index + 1)
        self.image_name_label.setText(f"画像: {relative.as_posix()}")
        width, height = self.image_sizes.get(relative, (0, 0))
        self.image_resolution_label.setText(f"解像度: {width} × {height}")
        self._loading_image = False
        self.statusBar().showMessage(
            f"{index + 1}/{len(self.images)}  {relative.as_posix()}  "
            f"BBox: {len(self.annotations.get(relative, []))}"
        )

    def _current_relative(self) -> Path | None:
        if 0 <= self.current_image_index < len(self.images):
            return self.images[self.current_image_index]
        return None

    # ---------- annotation editing ----------

    def begin_new_bbox(self) -> None:
        if self._current_relative() is None:
            return
        self.canvas.begin_new_bbox()
        self.statusBar().showMessage("画像上をドラッグして新しいBBoxを作成してください（Escで解除）")

    def _bbox_created(self, bbox: BBox) -> None:
        relative = self._current_relative()
        if relative is None:
            return
        class_id = self.label_combo.currentData()
        if class_id is None:
            class_id = 0
        certain = self.certain_check.isChecked()
        secondary_class_id = self.secondary_label_combo.currentData()
        annotation = Annotation(
            int(class_id),
            *bbox,
            secondary_class_id=(
                int(class_id)
                if certain
                else -1
                if secondary_class_id is None
                else int(secondary_class_id)
            ),
            certain=certain,
            visible=self.visible_check.isChecked(),
        )
        items = self.annotations.setdefault(relative, [])
        items.append(annotation)
        self.selected_annotation_index = len(items) - 1
        self._mark_dirty(relative)
        self.canvas.set_annotations(items, self.classes, self.selected_annotation_index)
        self._refresh_bbox_table()
        self._select_bbox_row(self.selected_annotation_index)
        self._load_annotation_form(self.selected_annotation_index)

    def _bbox_changed(self, index: int, bbox: BBox) -> None:
        relative = self._current_relative()
        if relative is None:
            return
        items = self.annotations.get(relative, [])
        if not 0 <= index < len(items):
            return
        items[index] = items[index].with_bbox(bbox)
        self._mark_dirty(relative)
        self.canvas.set_annotations(items, self.classes, index)
        self._refresh_bbox_table()
        self._select_bbox_row(index)
        self._load_annotation_form(index)

    def delete_selected_bbox(self) -> None:
        relative = self._current_relative()
        index = self.selected_annotation_index
        if relative is None or not 0 <= index < len(self.annotations.get(relative, [])):
            self.statusBar().showMessage("削除するBBoxを選択してください")
            return
        del self.annotations[relative][index]
        remaining = self.annotations[relative]
        new_index = min(index, len(remaining) - 1)
        self.selected_annotation_index = new_index
        self._mark_dirty(relative)
        self.canvas.set_annotations(remaining, self.classes, new_index)
        self._refresh_bbox_table()
        self._select_bbox_row(new_index)
        self._load_annotation_form(new_index)

    def _canvas_selection_changed(self, index: int) -> None:
        self.selected_annotation_index = index
        self._select_bbox_row(index)
        self._load_annotation_form(index)

    def _bbox_table_selection_changed(self) -> None:
        if self._loading_form:
            return
        rows = self.bbox_table.selectionModel().selectedRows()
        index = rows[0].row() if rows else -1
        self.selected_annotation_index = index
        self.canvas.set_selected_index(index)
        self._load_annotation_form(index)

    def _select_bbox_row(self, index: int) -> None:
        self._loading_form = True
        try:
            self.bbox_table.clearSelection()
            if 0 <= index < self.bbox_table.rowCount():
                self.bbox_table.selectRow(index)
        finally:
            self._loading_form = False

    def _load_annotation_form(self, index: int) -> None:
        relative = self._current_relative()
        items = self.annotations.get(relative, []) if relative else []
        enabled = 0 <= index < len(items)
        self._loading_form = True
        try:
            self.label_combo.setEnabled(bool(self.classes))
            self.certain_check.setEnabled(True)
            self.visible_check.setEnabled(True)
            if enabled:
                annotation = items[index]
                combo_index = self.label_combo.findData(annotation.class_id)
                if combo_index >= 0:
                    self.label_combo.setCurrentIndex(combo_index)
                secondary_index = self.secondary_label_combo.findData(
                    annotation.secondary_class_id
                )
                self.secondary_label_combo.setCurrentIndex(
                    secondary_index if secondary_index >= 0 else 0
                )
                self.certain_check.setChecked(annotation.certain)
                self.secondary_label_combo.setEnabled(not annotation.certain)
                self.visible_check.setChecked(annotation.visible)
                self.current_bbox_label.setText(
                    f"BBox: ({annotation.x_min}, {annotation.y_min}) - "
                    f"({annotation.x_max}, {annotation.y_max})"
                )
            else:
                self.current_bbox_label.setText("BBox: 未選択")
                self.secondary_label_combo.setEnabled(
                    not self.certain_check.isChecked()
                )
        finally:
            self._loading_form = False

    def _certain_toggled(self, checked: bool) -> None:
        self.secondary_label_combo.setEnabled(not checked)
        if self._loading_form:
            return
        primary_class_id = self.label_combo.currentData()
        if checked:
            primary_index = self.secondary_label_combo.findData(primary_class_id)
            if primary_index >= 0:
                with QSignalBlocker(self.secondary_label_combo):
                    self.secondary_label_combo.setCurrentIndex(primary_index)
        elif self.secondary_label_combo.currentData() in {-1, primary_class_id}:
            for index in range(1, self.secondary_label_combo.count()):
                if self.secondary_label_combo.itemData(index) != primary_class_id:
                    with QSignalBlocker(self.secondary_label_combo):
                        self.secondary_label_combo.setCurrentIndex(index)
                    break
        self._form_changed()

    def _form_changed(self, *args) -> None:
        if self._loading_form:
            return
        relative = self._current_relative()
        index = self.selected_annotation_index
        items = self.annotations.get(relative, []) if relative else []
        if not 0 <= index < len(items):
            return
        previous = items[index]
        class_id = self.label_combo.currentData()
        primary_class_id = int(
            class_id if class_id is not None else previous.class_id
        )
        certain = self.certain_check.isChecked()
        secondary_class_id = self.secondary_label_combo.currentData()
        if certain:
            secondary_index = self.secondary_label_combo.findData(primary_class_id)
            if (
                secondary_index >= 0
                and self.secondary_label_combo.currentIndex() != secondary_index
            ):
                with QSignalBlocker(self.secondary_label_combo):
                    self.secondary_label_combo.setCurrentIndex(secondary_index)
        updated = Annotation(
            primary_class_id,
            *previous.bbox,
            secondary_class_id=(
                primary_class_id
                if certain
                else -1
                if secondary_class_id is None
                else int(secondary_class_id)
            ),
            certain=certain,
            visible=self.visible_check.isChecked(),
        )
        if updated == previous:
            return
        items[index] = updated
        self._mark_dirty(relative)
        self.canvas.set_annotations(items, self.classes, index)
        self._refresh_bbox_table()
        self._select_bbox_row(index)

    def _mark_dirty(self, relative: Path) -> None:
        self.dirty.add(relative)
        self._update_image_table_row(relative)
        self.statusBar().showMessage(f"変更あり: {relative.as_posix()}")

    # ---------- tables ----------

    @staticmethod
    def _item(text: object, *, centered: bool = False) -> QTableWidgetItem:
        item = QTableWidgetItem(str(text))
        if centered:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"class_{class_id}"

    def _refresh_bbox_table(self) -> None:
        relative = self._current_relative()
        items = self.annotations.get(relative, []) if relative else []
        self._loading_form = True
        try:
            self.bbox_table.setRowCount(len(items))
            for row, annotation in enumerate(items):
                values = (
                    row + 1,
                    f"{annotation.class_id}: {self._class_name(annotation.class_id)}",
                    (
                        "-1: なし"
                        if annotation.secondary_class_id < 0
                        else f"{annotation.secondary_class_id}: "
                        f"{self._class_name(annotation.secondary_class_id)}"
                    ),
                    "1" if annotation.certain else "0",
                    "1" if annotation.visible else "0",
                    f"{annotation.x_min},{annotation.y_min},{annotation.x_max},{annotation.y_max}",
                )
                for column, value in enumerate(values):
                    self.bbox_table.setItem(
                        row,
                        column,
                        self._item(value, centered=column in {0, 2, 3, 4}),
                    )
        finally:
            self._loading_form = False

    def _populate_image_table(self) -> None:
        self.image_table.setRowCount(len(self.images))
        for row, relative in enumerate(self.images):
            self.image_table.setItem(row, 0, self._item(row + 1, centered=True))
            self.image_table.setItem(row, 1, self._item(relative.as_posix()))
            self._update_image_table_row(relative)

    def _update_image_table_row(self, relative: Path) -> None:
        try:
            row = self.images.index(relative)
        except ValueError:
            return
        self.image_table.setItem(
            row, 2, self._item(len(self.annotations.get(relative, [])), centered=True)
        )
        if relative in self.dirty:
            state = "未保存"
        elif relative in self.reviewed:
            state = "保存済み"
        else:
            state = "入力label"
        self.image_table.setItem(row, 3, self._item(state, centered=True))

    def _image_table_activated(self, row: int, column: int) -> None:
        self._show_image(row)

    # ---------- output ----------

    def _save_one(self, relative: Path) -> bool:
        if self.output_dir is None:
            QMessageBox.warning(self, "出力エラー", "出力先フォルダを選択してください。")
            return False
        size = self.image_sizes.get(relative)
        if not size or size[0] <= 0 or size[1] <= 0:
            QMessageBox.warning(self, "出力エラー", f"画像サイズが不明です: {relative}")
            return False
        label_relative = annotation_relative_path(relative)
        try:
            save_annotations(
                self.output_dir / "labels" / label_relative,
                self.annotations.get(relative, []),
                *size,
                extended=False,
            )
            save_annotations(
                self.output_dir / "labels_extended" / label_relative,
                self.annotations.get(relative, []),
                *size,
                extended=True,
            )
            save_classes(self.output_dir, self.classes)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "保存エラー", f"{relative} の保存に失敗しました。\n{exc}")
            return False
        self.dirty.discard(relative)
        self.reviewed.add(relative)
        self._update_image_table_row(relative)
        return True

    def save_all(self, *, show_dialog: bool = True) -> bool:
        if not self.images:
            if show_dialog:
                QMessageBox.information(self, "保存", "データセットが読み込まれていません。")
            return False
        failures: list[Path] = []
        for relative in self.images:
            if not self._save_one(relative):
                failures.append(relative)
                break
        if failures:
            return False
        total_boxes = sum(len(items) for items in self.annotations.values())
        self.dataset_summary_label.setText(
            f"画像 {len(self.images)}枚 / BBox {total_boxes}個 / "
            f"出力済み {len(self.reviewed)}枚 / 警告 {len(self.load_errors)}件"
        )
        self.statusBar().showMessage(
            f"保存完了: {self.output_dir}（画像 {len(self.images)}枚 / BBox {total_boxes}個）"
        )
        if show_dialog:
            QMessageBox.information(
                self,
                "保存完了",
                f"{len(self.images)}画像・{total_boxes} BBoxを保存しました。\n\n"
                f"通常YOLO: {self.output_dir / 'labels'}\n"
                f"属性付き: {self.output_dir / 'labels_extended'}",
            )
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if not self.dirty:
            event.accept()
            return
        response = QMessageBox.question(
            self,
            "終了確認",
            "未保存の変更があります。保存して終了しますか？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if response == QMessageBox.StandardButton.Cancel:
            event.ignore()
        elif response == QMessageBox.StandardButton.Save:
            event.accept() if self.save_all(show_dialog=False) else event.ignore()
        else:
            event.accept()
