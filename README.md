# YOLO BBox Annotator

データセット内の `images/`、既存YOLO形式の `labels/`、`classes.txt` を読み込み、画像ごとにBBoxと追加属性を編集する
PySide6デスクトップアプリです。`flower_gt_annotator` の画面構成とBBox操作を参考にしています。

このワークスペースでは、起動時に `../datasets_refresh300` を自動検出します。
入力データは変更せず、既定では `datasets_refresh300/annotation_output` へ結果を保存します。

## 主な機能

- 画像と同名の既存YOLOラベルを読み込み、全BBoxを重ねて表示
- 空き領域のドラッグ、または「新規BBox」後のドラッグでBBoxを追加
- BBoxクリックで選択、内部ドラッグで移動、四隅ドラッグでリサイズ
- BBox単位で次の情報を編集
  - `label`: `classes.txt` のクラス
  - `第2 label`: certain=0の場合の第2候補クラス。certain=1の場合はlabelと同じクラス
  - `certain`: 0/1
  - `visible`: 0/1
- 前後画像への移動時に変更を自動保存
- 画像一覧からダブルクリックで移動
- 通常YOLO形式と属性付き形式を別フォルダへ同時出力
- 以前の出力があれば属性を含めて作業を再開

## セットアップと起動

Linux:

```bash
cd yolo_bbox_annotator
chmod +x run_linux.sh
./run_linux.sh
```

Windows:

```bat
cd yolo_bbox_annotator
run_windows.bat
```

手動起動:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

別データセットを指定する場合:

```bash
python main.py \
  --dataset-dir /path/to/dataset
```

入力フォルダは次の構成を前提とします。

```text
dataset/
├── images/
├── labels/
└── classes.txt
```

## 出力

「全画像を保存」を実行すると、画像と同じ相対パス・ベース名で全画像分のTXTを出力します。
BBoxが0個の画像には空のTXTを出力します。

```text
dataset/
├── images/
├── labels/
├── classes.txt
└── annotation_output/
    ├── classes.txt
    ├── labels/
    │   └── 1013_0011.txt
    └── labels_extended/
        └── 1013_0011.txt
```

`labels/` は一般的なYOLO形式です。

```text
class_id x_center y_center width height
```

`labels_extended/` は同じ5列の後ろに属性を追加します。

```text
class_id x_center y_center width height secondary_class_id is_certain is_visible
```

- 座標4列は0〜1の正規化値
- `secondary_class_id` は第2クラスID。`is_certain=1` の場合は `class_id` と同じ値
- `is_certain` と `is_visible` は `0` または `1`

再開時は `labels_extended/`、出力済み `labels/`、入力 `labels/` の順に読み込みます。

## 操作

| 操作 | 動作 |
|---|---|
| 空き領域を左ドラッグ | 新規BBox |
| BBoxをクリック | 選択 |
| 選択BBoxの内部をドラッグ | 移動 |
| 選択BBoxの四隅をドラッグ | リサイズ |
| Ctrl+マウスホイール | カーソル位置を中心に拡大 / 縮小（最小は全体表示） |
| Ctrl+0 | 拡大を解除して全体表示 |
| Ctrl+N | 新規BBoxモード |
| Ctrl+D | 選択BBoxを削除 |
| Ctrl+Left / Ctrl+Right | 前 / 次の画像 |
| Ctrl+S | 全画像を両形式で保存 |
| Esc | 新規BBoxモードを解除 |

## テスト

```bash
cd yolo_bbox_annotator
python -m unittest discover -s tests -v
```
