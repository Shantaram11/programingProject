import json
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
import os
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet

from core.pipeline import (
    AvailableModels,
    CleaningConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    TrainingResult,
    load_dataset,
    run_training,
)
from core.storage import RunStore

try:
    from openai import OpenAI
except Exception:  # optional at runtime if user didn't install deps
    OpenAI = None  # type: ignore[assignment]

APP_NAME = "Time-Series Model Training Platform"

MODEL_LABELS: Dict[str, str] = {
    "ma": "MA (Moving Average)",
    "wma": "WMA (Weighted Moving Average)",
    "arima": "ARIMA (AutoRegressive Integrated Moving Average)",
    "prophet": "PROPHET (Facebook/Meta Prophet)",
    "xgboost": "XGBoost",
    "deepar": "DeepAR (Deep Autoregressive Recurrent Network)",
}

EVAL_METRIC_ITEMS: List[tuple[str, str]] = [
    ("RMSE (Root Mean Squared Error)", "rmse"),
    ("MAE (Mean Absolute Error)", "mae"),
    ("MAPE% (Mean Absolute Percentage Error)", "mape_pct"),
    ("MSE (Mean Squared Error)", "mse"),
    ("MaxError (Max Absolute Error)", "max_error"),
    ("MinError (Min Absolute Error)", "min_error"),
]

EVAL_METRIC_DEFAULTS = {"rmse", "mae", "mape_pct"}

MISSING_METHOD_ITEMS: List[tuple[str, str]] = [
    ("No autofill", "none"),
    ("Drop rows with missing values", "drop_rows"),
    ("Forward fill", "ffill"),
    ("Backward fill", "bfill"),
    ("Linear interpolation", "interpolate_linear"),
]

OUTLIER_METHOD_ITEMS: List[tuple[str, str]] = [
    ("No outlier handling", "none"),
    ("Quantile clipping", "clip_quantile"),
]

SCALER_ITEMS: List[tuple[str, str]] = [
    ("No scaling", "none"),
    ("StandardScaler", "standard"),
    ("MinMaxScaler", "minmax"),
]

TARGET_TRANSFORM_ITEMS: List[tuple[str, str]] = [
    ("No transform", "none"),
    ("llog(1+x)", "log1p"),
]


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify_name(name: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name.strip())
    s = "_".join([p for p in s.split("_") if p])  # collapse repeats
    return s[:64]


def _human_ts(ts: str) -> str:
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


class MplCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        fig = Figure(figsize=(9, 5), tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class CheckBoxList(QWidget):
    """A scrollable list of explicit QCheckBox widgets (qt-material friendly)."""

    def __init__(self, checked_by_default: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checked_by_default = checked_by_default
        self._boxes: Dict[str, QCheckBox] = {}

        outer = QVBoxLayout()
        self.setLayout(outer)
        # QFormLayout can collapse custom widgets; enforce a reasonable default height.
        self.setMinimumHeight(220)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(220)
        self._scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._scroll, 1)

        inner = QWidget()
        self._v = QVBoxLayout()
        inner.setLayout(self._v)
        self._scroll.setWidget(inner)

        self._v.addStretch(1)

    def set_items(self, items: List[str]) -> None:
        # clear existing
        self._boxes.clear()
        while self._v.count():
            it = self._v.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

        for name in items:
            cb = QCheckBox(name)
            cb.setChecked(self._checked_by_default)
            self._v.addWidget(cb)
            self._boxes[name] = cb
        self._v.addStretch(1)

    def checked_items(self) -> List[str]:
        return [k for k, cb in self._boxes.items() if cb.isChecked()]

    def set_checked(self, name: str, checked: bool) -> None:
        if name in self._boxes:
            self._boxes[name].setChecked(checked)


class _NoWheelMixin:
    """Prevent accidental value changes from mouse wheel scrolling."""

    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


class NoWheelSpinBox(_NoWheelMixin, QSpinBox):
    pass


class NoWheelDoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    pass


class NoWheelComboBox(_NoWheelMixin, QComboBox):
    pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 850)

        self.store = RunStore.default()

        self.df: Optional[pd.DataFrame] = None
        self.df_path: Optional[str] = None
        self.last_result: Optional[TrainingResult] = None
        self._gpt_messages: List[Dict[str, str]] = []
        self._reset_gpt_memory()

        self._build_ui()
        self._refresh_saved_runs()
        self._refresh_model_availability_badges()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.setCentralWidget(tabs)

        self.tab_data = QWidget()
        self.tab_clean = QWidget()
        self.tab_models = QWidget()
        self.tab_saved = QWidget()

        tabs.addTab(self.tab_data, "1) Data")
        tabs.addTab(self.tab_clean, "2) Cleaning & Features")
        tabs.addTab(self.tab_models, "3) Models, Train & Visualize")
        tabs.addTab(self.tab_saved, "4) Saved Results")

        self._build_data_tab()
        self._build_clean_tab()
        self._build_models_tab()
        self._build_saved_tab()

    # -------------------------
    # Tab 1: Data
    # -------------------------
    def _build_data_tab(self) -> None:
        layout = QVBoxLayout()
        self.tab_data.setLayout(layout)

        top = QHBoxLayout()
        layout.addLayout(top)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Choose a CSV file")
        self.file_path_edit.setReadOnly(True)
        top.addWidget(self.file_path_edit, 1)

        btn_pick = QPushButton("Upload data")
        btn_pick.clicked.connect(self._pick_file)
        top.addWidget(btn_pick)

        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._load_file)
        top.addWidget(btn_load)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Left: preview
        left = QWidget()
        left_layout = QVBoxLayout()
        left.setLayout(left_layout)
        splitter.addWidget(left)

        left_layout.addWidget(QLabel("Preview (first 30 rows):"))
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        left_layout.addWidget(self.preview_table, 1)

        # Right: column selections
        right = QWidget()
        right_layout = QVBoxLayout()
        right.setLayout(right_layout)
        splitter.addWidget(right)

        group = QGroupBox("Columns")
        form = QFormLayout()
        group.setLayout(form)
        right_layout.addWidget(group)

        self.time_col_combo = NoWheelComboBox()
        self.time_col_combo.setToolTip("Datetime column. Range: any datetime-like column.")
        form.addRow(QLabel("Time column"), self.time_col_combo)

        self.freq_combo = NoWheelComboBox()
        self.freq_combo.addItems(["auto", "D", "H", "T", "S", "W", "M"])
        self.freq_combo.setToolTip("Data frequency. Default: auto. Options: D/H/T/S/W/M.")
        form.addRow(QLabel("Frequency"), self.freq_combo)

        self.sort_time_chk = QCheckBox("Sort by time ascending")
        self.sort_time_chk.setChecked(True)
        self.sort_time_chk.setToolTip("Default: on.")
        form.addRow(QLabel(""), self.sort_time_chk)

        # replaced by explicit checkbox list to ensure visibility under qt-material
        self.targets_checks = CheckBoxList(checked_by_default=False)
        self.targets_checks.setToolTip("Target variables to predict. Default: none selected.")
        form.addRow(QLabel("Targets (multi-select)"), self.targets_checks)

        self.features_checks = CheckBoxList(checked_by_default=True)
        self.features_checks.setToolTip("Optional feature columns. Default: all numeric (excluding targets).")
        form.addRow(QLabel("Features (multi-select)"), self.features_checks)

        group2 = QGroupBox("Train/Test")
        form2 = QFormLayout()
        group2.setLayout(form2)
        right_layout.addWidget(group2)

        self.train_ratio = NoWheelDoubleSpinBox()
        self.train_ratio.setRange(0.5, 0.95)
        self.train_ratio.setSingleStep(0.05)
        self.train_ratio.setValue(0.8)
        self.train_ratio.setToolTip("Train ratio. Default: 0.80. Range: 0.50–0.95.")
        form2.addRow(QLabel("Train ratio"), self.train_ratio)

        self.horizon = NoWheelSpinBox()
        self.horizon.setRange(1, 5000)
        self.horizon.setValue(24)
        self.horizon.setToolTip("Forecast horizon. Default: 24. Range: 1–5000.")
        form2.addRow(QLabel("Forecast horizon"), self.horizon)

        self.lag_window = NoWheelSpinBox()
        self.lag_window.setRange(1, 5000)
        self.lag_window.setValue(48)
        self.lag_window.setToolTip("Lag window (ML/Deep models). Default: 48. Range: 1–5000.")
        form2.addRow(QLabel("Lag window"), self.lag_window)

        hint = QLabel(
            ""
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("opacity: 0.9;")
        right_layout.addWidget(hint)
        right_layout.addStretch(1)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # no extra time builder UI

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose dataset file",
            str(Path.home()),
            "Data files (*.csv *.xlsx *.xls *.parquet);;All files (*.*)",
        )
        if not path:
            return
        self.file_path_edit.setText(path)
        self.df_path = path

    def _load_file(self) -> None:
        if not self.df_path:
            QMessageBox.warning(self, "No file", "Please upload a dataset file first.")
            return
        try:
            self.df = load_dataset(self.df_path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{e}\n\n{traceback.format_exc()}")
            return

        self._populate_preview()
        self._populate_column_selectors()
        QMessageBox.information(self, "Loaded", f"Loaded dataset with shape: {self.df.shape}")

    def _populate_preview(self) -> None:
        assert self.df is not None
        view = self.df.head(30)
        self.preview_table.setRowCount(len(view))
        self.preview_table.setColumnCount(len(view.columns))
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in view.columns])
        for r in range(len(view)):
            for c, col in enumerate(view.columns):
                item = QTableWidgetItem(str(view.iloc[r, c]))
                self.preview_table.setItem(r, c, item)
        self.preview_table.resizeColumnsToContents()

    def _populate_column_selectors(self) -> None:
        assert self.df is not None
        self.time_col_combo.clear()
        self.targets_checks.set_items([])
        self.features_checks.set_items([])

        # Ignore common "Unnamed: 0" index columns and blank column names.
        cols = [
            c
            for c in list(self.df.columns)
            if str(c).strip() != "" and not str(c).strip().lower().startswith("unnamed")
        ]
        # Add a "None" option to use row order as time
        self.time_col_combo.addItem("None (use row order)", "__ts__")
        for c in cols:
            self.time_col_combo.addItem(str(c), str(c))

        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
        self.targets_checks.set_items([str(c) for c in numeric_cols])
        self.features_checks.set_items([str(c) for c in numeric_cols])

    # -------------------------
    # Tab 2: Cleaning
    # -------------------------
    def _build_clean_tab(self) -> None:
        layout = QVBoxLayout()
        self.tab_clean.setLayout(layout)

        group = QGroupBox("Data Cleaning")
        form = QFormLayout()
        group.setLayout(form)
        layout.addWidget(group)

        self.missing_method = NoWheelComboBox()
        for label, code in MISSING_METHOD_ITEMS:
            self.missing_method.addItem(label, code)
        # Default: ffill
        for i in range(self.missing_method.count()):
            if self.missing_method.itemData(i) == "ffill":
                self.missing_method.setCurrentIndex(i)
                break
        self.missing_method.setToolTip(
            "Missing value handling. Default: ffill (Forward fill). "
            "Options: none/drop_rows/ffill/bfill/interpolate_linear."
        )
        form.addRow(QLabel("Missing values"), self.missing_method)

        self.outlier_method = NoWheelComboBox()
        for label, code in OUTLIER_METHOD_ITEMS:
            self.outlier_method.addItem(label, code)
        for i in range(self.outlier_method.count()):
            if self.outlier_method.itemData(i) == "none":
                self.outlier_method.setCurrentIndex(i)
                break
        self.outlier_method.setToolTip("Outlier handling. Default: none. Options: none/clip_quantile.")
        form.addRow(QLabel("Outliers"), self.outlier_method)

        qrow = QHBoxLayout()
        self.q_low = NoWheelDoubleSpinBox()
        self.q_low.setDecimals(3)
        self.q_low.setRange(0.0, 0.49)
        self.q_low.setValue(0.01)
        self.q_low.setSingleStep(0.01)
        self.q_low.setToolTip("Lower quantile. Default: 0.01. Range: 0.00–0.49.")
        self.q_high = NoWheelDoubleSpinBox()
        self.q_high.setDecimals(3)
        self.q_high.setRange(0.51, 1.0)
        self.q_high.setValue(0.99)
        self.q_high.setSingleStep(0.01)
        self.q_high.setToolTip("Upper quantile. Default: 0.99. Range: 0.51–1.00.")
        qrow.addWidget(QLabel("low"))
        qrow.addWidget(self.q_low)
        qrow.addSpacing(12)
        qrow.addWidget(QLabel("high"))
        qrow.addWidget(self.q_high)
        qwrap = QWidget()
        qwrap.setLayout(qrow)
        form.addRow(QLabel("Clip quantiles"), qwrap)

        self.scaler = NoWheelComboBox()
        for label, code in SCALER_ITEMS:
            self.scaler.addItem(label, code)
        for i in range(self.scaler.count()):
            if self.scaler.itemData(i) == "standard":
                self.scaler.setCurrentIndex(i)
                break
        self.scaler.setToolTip("Scaling for ML/Deep models. Default: standard. Options: none/standard/minmax.")
        form.addRow(QLabel("Scaling"), self.scaler)

        self.transform = NoWheelComboBox()
        for label, code in TARGET_TRANSFORM_ITEMS:
            self.transform.addItem(label, code)
        for i in range(self.transform.count()):
            if self.transform.itemData(i) == "none":
                self.transform.setCurrentIndex(i)
                break
        self.transform.setToolTip("Target transform. Default: none. Options: none/log1p.")
        form.addRow(QLabel("Target transform"), self.transform)

        layout.addStretch(1)

    # -------------------------
    # Tab 3: Models, Train & Visualize
    # -------------------------
    def _build_models_tab(self) -> None:
        layout = QHBoxLayout()
        self.tab_models.setLayout(layout)

        left = QWidget()
        left_layout = QVBoxLayout()
        left.setLayout(left_layout)
        layout.addWidget(left, 1)

        # Models controls inside scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll, 1)

        scroll_inner = QWidget()
        self.models_form = QVBoxLayout()
        scroll_inner.setLayout(self.models_form)
        scroll.setWidget(scroll_inner)

        self.model_widgets: Dict[str, Dict[str, Any]] = {}
        self._add_model_group_ma()
        self._add_model_group_wma()
        self._add_model_group_arima()
        self._add_model_group_prophet()
        self._add_model_group_xgb()
        self._add_model_group_deepar()
        self.models_form.addStretch(1)

        # Actions
        actions = QHBoxLayout()
        left_layout.addLayout(actions)

        btn_train = QPushButton("Train selected models")
        btn_train.clicked.connect(self._train)
        actions.addWidget(btn_train, 1)

        btn_save = QPushButton("Save last result")
        btn_save.clicked.connect(self._save_last_result)
        actions.addWidget(btn_save)

        # Logs + GPT suggestions + clear button
        out_toolbar = QHBoxLayout()
        left_layout.addLayout(out_toolbar)
        btn_clear_hist = QPushButton("Clear history (logs + GPT)")
        btn_clear_hist.clicked.connect(self._clear_history_and_gpt)
        out_toolbar.addWidget(btn_clear_hist)
        out_toolbar.addStretch(1)

        # OpenAI key controls (process environment)
        gpt_toggle_row = QHBoxLayout()
        left_layout.addLayout(gpt_toggle_row)
        self.enable_openai_chk = QCheckBox("Enable OpenAI training advisor")
        self.enable_openai_chk.setChecked(False)
        self.enable_openai_chk.setToolTip("If enabled, the app will call OpenAI after training to suggest improvements.")
        self.enable_openai_chk.toggled.connect(self._on_openai_toggle)
        gpt_toggle_row.addWidget(self.enable_openai_chk)
        gpt_toggle_row.addStretch(1)

        key_row = QHBoxLayout()
        left_layout.addLayout(key_row)
        key_row.addWidget(QLabel("OpenAI API key:"))
        self.openai_key_edit = QLineEdit()
        self.openai_key_edit.setEchoMode(QLineEdit.Password)
        self.openai_key_edit.setPlaceholderText("Paste your OPENAI_API_KEY here…")
        key_row.addWidget(self.openai_key_edit, 1)
        btn_set_key = QPushButton("Set key for this app session")
        btn_set_key.clicked.connect(self._set_openai_key_from_ui)
        key_row.addWidget(btn_set_key)
        self._openai_key_widgets = [self.openai_key_edit, btn_set_key]

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText(
            "Output will appear here…\n"
            "- Training log\n"
            "- GPT training suggestions (if enabled)\n"
        )
        left_layout.addWidget(self.log, 1)
        # Now safe (log exists)
        self._on_openai_toggle(False)

        # Right: visualization
        right = QWidget()
        right_layout = QVBoxLayout()
        right.setLayout(right_layout)
        layout.addWidget(right, 1)

        top_controls = QHBoxLayout()
        right_layout.addLayout(top_controls)

        self.target_view_combo = NoWheelComboBox()
        self.target_view_combo.currentTextChanged.connect(self._redraw_plot)
        top_controls.addWidget(QLabel("Target:"))
        top_controls.addWidget(self.target_view_combo, 1)

        self.model_lines_box = QGroupBox("Show/Hide model lines")
        self.model_lines_layout = QVBoxLayout()
        self.model_lines_box.setLayout(self.model_lines_layout)
        right_layout.addWidget(self.model_lines_box)

        fig_toolbar = QHBoxLayout()
        btn_save_fig = QPushButton("Save plot…")
        btn_save_fig.clicked.connect(self._save_current_plot)
        fig_toolbar.addWidget(btn_save_fig)
        fig_toolbar.addStretch(1)
        right_layout.addLayout(fig_toolbar)

        self.canvas = MplCanvas()
        right_layout.addWidget(self.canvas, 1)

        self.eval_metrics_box = QGroupBox("Evaluation metrics (multi-select)")
        em_layout = QVBoxLayout()
        self.eval_metrics_box.setLayout(em_layout)
        self._eval_metric_checks: Dict[str, QCheckBox] = {}
        for label, code in EVAL_METRIC_ITEMS:
            chk = QCheckBox(label)
            chk.setChecked(code in EVAL_METRIC_DEFAULTS)
            em_layout.addWidget(chk)
            self._eval_metric_checks[code] = chk
        em_layout.addStretch(1)
        right_layout.addWidget(self.eval_metrics_box)

        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        right_layout.addWidget(QLabel("Metrics (per model):"))
        right_layout.addWidget(self.metrics_table, 1)

    def _add_model_group_common(self, title: str, key: str, enabled_default: bool) -> QGroupBox:
        box = QGroupBox(title)
        # Using an explicit checkbox because qt-material can hide the groupbox check indicator.
        box.setToolTip("Model settings.")
        self.models_form.addWidget(box)
        form = QFormLayout()
        box.setLayout(form)
        enabled = QCheckBox("Enable this model")
        enabled.setChecked(enabled_default)
        enabled.setToolTip("Include/exclude this model. Default shown by checkbox state.")
        form.addRow(QLabel(""), enabled)

        self.model_widgets[key] = {"box": box, "enabled": enabled, "form": form, "availability": QLabel("")}
        badge = self.model_widgets[key]["availability"]
        badge.setStyleSheet("opacity: 0.85;")
        form.addRow(QLabel("Availability"), badge)
        return box

    def _add_model_group_ma(self) -> None:
        self._add_model_group_common(MODEL_LABELS["ma"], "ma", enabled_default=True)
        form: QFormLayout = self.model_widgets["ma"]["form"]

        w = NoWheelSpinBox()
        w.setRange(1, 5000)
        w.setValue(24)
        w.setToolTip("MA window. Default: 24. Range: 1–5000.")
        form.addRow(QLabel("window"), w)
        self.model_widgets["ma"]["window"] = w

    def _add_model_group_wma(self) -> None:
        self._add_model_group_common(MODEL_LABELS["wma"], "wma", enabled_default=False)
        form: QFormLayout = self.model_widgets["wma"]["form"]

        w = NoWheelSpinBox()
        w.setRange(1, 5000)
        w.setValue(24)
        w.setToolTip("WMA window. Default: 24. Range: 1–5000.")
        form.addRow(QLabel("window"), w)
        self.model_widgets["wma"]["window"] = w

        scheme = NoWheelComboBox()
        scheme.addItems(["linear_recent_heavier", "linear_older_heavier"])
        scheme.setCurrentText("linear_recent_heavier")
        scheme.setToolTip("Weight scheme. Default: linear_recent_heavier.")
        form.addRow(QLabel("weights"), scheme)
        self.model_widgets["wma"]["weights"] = scheme

    def _add_model_group_arima(self) -> None:
        self._add_model_group_common(MODEL_LABELS["arima"], "arima", enabled_default=False)
        form: QFormLayout = self.model_widgets["arima"]["form"]

        p = NoWheelSpinBox()
        p.setRange(0, 10)
        p.setValue(2)
        p.setToolTip("p order. Default: 2. Range: 0–10.")
        d = NoWheelSpinBox()
        d.setRange(0, 3)
        d.setValue(1)
        d.setToolTip("d order. Default: 1. Range: 0–3.")
        q = NoWheelSpinBox()
        q.setRange(0, 10)
        q.setValue(2)
        q.setToolTip("q order. Default: 2. Range: 0–10.")

        row = QHBoxLayout()
        row.addWidget(QLabel("p"))
        row.addWidget(p)
        row.addSpacing(8)
        row.addWidget(QLabel("d"))
        row.addWidget(d)
        row.addSpacing(8)
        row.addWidget(QLabel("q"))
        row.addWidget(q)
        w = QWidget()
        w.setLayout(row)
        form.addRow(QLabel("order"), w)
        self.model_widgets["arima"]["p"] = p
        self.model_widgets["arima"]["d"] = d
        self.model_widgets["arima"]["q"] = q

        use_exog = QCheckBox("Use selected feature columns as exogenous variables (if any)")
        use_exog.setChecked(False)
        use_exog.setToolTip("Default: off.")
        form.addRow(QLabel(""), use_exog)
        self.model_widgets["arima"]["use_exog"] = use_exog

    def _add_model_group_prophet(self) -> None:
        self._add_model_group_common(MODEL_LABELS["prophet"], "prophet", enabled_default=False)
        form: QFormLayout = self.model_widgets["prophet"]["form"]

        cp = NoWheelDoubleSpinBox()
        cp.setDecimals(4)
        cp.setRange(0.001, 2.0)
        cp.setValue(0.05)
        cp.setSingleStep(0.01)
        cp.setToolTip("changepoint_prior_scale. Default: 0.05. Range: 0.001–2.0.")
        form.addRow(QLabel("changepoint_prior_scale"), cp)
        self.model_widgets["prophet"]["cp"] = cp

        sp = NoWheelDoubleSpinBox()
        sp.setDecimals(4)
        sp.setRange(0.01, 20.0)
        sp.setValue(10.0)
        sp.setSingleStep(0.5)
        sp.setToolTip("seasonality_prior_scale. Default: 10.0. Range: 0.01–20.0.")
        form.addRow(QLabel("seasonality_prior_scale"), sp)
        self.model_widgets["prophet"]["sp"] = sp

        mode = NoWheelComboBox()
        mode.addItems(["additive", "multiplicative"])
        mode.setCurrentText("additive")
        mode.setToolTip("seasonality_mode. Default: additive.")
        form.addRow(QLabel("seasonality_mode"), mode)
        self.model_widgets["prophet"]["mode"] = mode

        ncp = NoWheelSpinBox()
        ncp.setRange(0, 100)
        ncp.setValue(25)
        ncp.setToolTip("n_changepoints. Default: 25. Range: 0–100.")
        form.addRow(QLabel("n_changepoints"), ncp)
        self.model_widgets["prophet"]["ncp"] = ncp

        use_regs = QCheckBox("Use selected feature columns as regressors (if any)")
        use_regs.setChecked(False)
        use_regs.setToolTip("Default: off.")
        form.addRow(QLabel(""), use_regs)
        self.model_widgets["prophet"]["use_regs"] = use_regs

    def _add_model_group_xgb(self) -> None:
        self._add_model_group_common(MODEL_LABELS["xgboost"], "xgboost", enabled_default=True)
        form: QFormLayout = self.model_widgets["xgboost"]["form"]

        depth = NoWheelSpinBox()
        depth.setRange(1, 20)
        depth.setValue(6)
        depth.setToolTip("max_depth. Default: 6. Range: 1–20.")
        form.addRow(QLabel("max_depth"), depth)
        self.model_widgets["xgboost"]["max_depth"] = depth

        lr = NoWheelDoubleSpinBox()
        lr.setDecimals(4)
        lr.setRange(0.0001, 1.0)
        lr.setValue(0.01)
        lr.setSingleStep(0.01)
        lr.setToolTip("learning_rate. Default: 0.01. Range: 0.0001–1.0.")
        form.addRow(QLabel("learning_rate"), lr)
        self.model_widgets["xgboost"]["learning_rate"] = lr

        n_estimators = NoWheelSpinBox()
        n_estimators.setRange(10, 5000)
        n_estimators.setValue(600)
        n_estimators.setToolTip("n_estimators. Default: 600. Range: 10–5000.")
        form.addRow(QLabel("n_estimators"), n_estimators)
        self.model_widgets["xgboost"]["n_estimators"] = n_estimators

        subsample = NoWheelDoubleSpinBox()
        subsample.setDecimals(3)
        subsample.setRange(0.2, 1.0)
        subsample.setValue(0.9)
        subsample.setSingleStep(0.05)
        subsample.setToolTip("subsample. Default: 0.9. Range: 0.2–1.0.")
        form.addRow(QLabel("subsample"), subsample)
        self.model_widgets["xgboost"]["subsample"] = subsample

        colsample = NoWheelDoubleSpinBox()
        colsample.setDecimals(3)
        colsample.setRange(0.2, 1.0)
        colsample.setValue(0.9)
        colsample.setSingleStep(0.05)
        colsample.setToolTip("colsample_bytree. Default: 0.9. Range: 0.2–1.0.")
        form.addRow(QLabel("colsample_bytree"), colsample)
        self.model_widgets["xgboost"]["colsample_bytree"] = colsample

    def _add_model_group_deepar(self) -> None:
        self._add_model_group_common(MODEL_LABELS["deepar"], "deepar", enabled_default=False)
        form: QFormLayout = self.model_widgets["deepar"]["form"]

        hidden = NoWheelSpinBox()
        hidden.setRange(8, 512)
        hidden.setValue(64)
        hidden.setToolTip("hidden_size. Default: 64. Range: 8–512.")
        form.addRow(QLabel("hidden_size"), hidden)
        self.model_widgets["deepar"]["hidden_size"] = hidden

        layers = NoWheelSpinBox()
        layers.setRange(1, 4)
        layers.setValue(2)
        layers.setToolTip("num_layers. Default: 2. Range: 1–4.")
        form.addRow(QLabel("num_layers"), layers)
        self.model_widgets["deepar"]["num_layers"] = layers

        dropout = NoWheelDoubleSpinBox()
        dropout.setDecimals(3)
        dropout.setRange(0.0, 0.8)
        dropout.setValue(0.1)
        dropout.setSingleStep(0.05)
        dropout.setToolTip("dropout. Default: 0.1. Range: 0.0–0.8.")
        form.addRow(QLabel("dropout"), dropout)
        self.model_widgets["deepar"]["dropout"] = dropout

        epochs = NoWheelSpinBox()
        epochs.setRange(1, 200)
        epochs.setValue(20)
        epochs.setToolTip("epochs. Default: 20. Range: 1–200.")
        form.addRow(QLabel("epochs"), epochs)
        self.model_widgets["deepar"]["epochs"] = epochs

        lr = NoWheelDoubleSpinBox()
        lr.setDecimals(5)
        lr.setRange(1e-5, 1e-1)
        lr.setValue(1e-3)
        lr.setSingleStep(5e-4)
        lr.setToolTip("learning_rate. Default: 0.001. Range: 1e-5–1e-1.")
        form.addRow(QLabel("learning_rate"), lr)
        self.model_widgets["deepar"]["learning_rate"] = lr

        batch = NoWheelSpinBox()
        batch.setRange(8, 1024)
        batch.setValue(64)
        batch.setToolTip("batch_size. Default: 64. Range: 8–1024.")
        form.addRow(QLabel("batch_size"), batch)
        self.model_widgets["deepar"]["batch_size"] = batch

        device = NoWheelComboBox()
        device.addItems(["auto", "cpu"])
        device.setCurrentText("auto")
        device.setToolTip("device. Default: auto. Options: auto/cpu.")
        form.addRow(QLabel("device"), device)
        self.model_widgets["deepar"]["device"] = device

    def _refresh_model_availability_badges(self) -> None:
        avail = AvailableModels.detect()
        for key, w in self.model_widgets.items():
            badge: QLabel = w["availability"]
            ok, msg = avail.status_for(key)
            badge.setText("✅ Available" if ok else f"⚠ Missing deps: {msg}")
            if not ok:
                # Allow user to still check it, but training will error with a clear message.
                badge.setStyleSheet("color: #d19a66;")

    def _append_log(self, msg: str) -> None:
        self.log.appendPlainText(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _collect_config(self) -> PipelineConfig:
        if self.df is None:
            raise ValueError("No dataset loaded. Go to 'Data' tab and load a file.")

        # Time column can be an actual column or "__ts__" meaning row order
        time_col = str(self.time_col_combo.currentData() or self.time_col_combo.currentText())
        targets = self.targets_checks.checked_items()
        if not targets:
            raise ValueError("Please select at least 1 target variable.")

        feats = self.features_checks.checked_items()
        # By default features list is all numeric; remove targets to avoid leakage unless user explicitly keeps them.
        feats = [c for c in feats if c not in targets]

        eval_metrics = [k for k, chk in self._eval_metric_checks.items() if chk.isChecked()]
        if not eval_metrics:
            raise ValueError("Please select at least 1 evaluation metric (e.g. RMSE, MAE, MSE).")

        cleaning = CleaningConfig(
            missing_method=str(self.missing_method.currentData() or "ffill"),
            outlier_method=str(self.outlier_method.currentData() or "none"),
            clip_q_low=float(self.q_low.value()),
            clip_q_high=float(self.q_high.value()),
            scaler=str(self.scaler.currentData() or "standard"),
            target_transform=str(self.transform.currentData() or "none"),
        )

        features = FeatureConfig(
            time_col=time_col,
            targets=targets,
            features=feats,
            freq=self.freq_combo.currentText(),
            sort_time=bool(self.sort_time_chk.isChecked()),
            train_ratio=float(self.train_ratio.value()),
            horizon=int(self.horizon.value()),
            lag_window=int(self.lag_window.value()),
            eval_metrics=eval_metrics,
        )

        models: Dict[str, ModelConfig] = {}
        for key, w in self.model_widgets.items():
            enabled: QCheckBox = w["enabled"]
            if not enabled.isChecked():
                continue
            params: Dict[str, Any] = {}
            if key == "ma":
                params["window"] = int(w["window"].value())
            elif key == "wma":
                params["window"] = int(w["window"].value())
                params["weights"] = str(w["weights"].currentText())
            elif key == "arima":
                params["p"] = int(w["p"].value())
                params["d"] = int(w["d"].value())
                params["q"] = int(w["q"].value())
                params["use_exog"] = bool(w["use_exog"].isChecked())
            elif key == "prophet":
                params["changepoint_prior_scale"] = float(w["cp"].value())
                params["seasonality_prior_scale"] = float(w["sp"].value())
                params["seasonality_mode"] = str(w["mode"].currentText())
                params["n_changepoints"] = int(w["ncp"].value())
                params["use_regressors"] = bool(w["use_regs"].isChecked())
            elif key == "xgboost":
                params["max_depth"] = int(w["max_depth"].value())
                params["learning_rate"] = float(w["learning_rate"].value())
                params["n_estimators"] = int(w["n_estimators"].value())
                params["subsample"] = float(w["subsample"].value())
                params["colsample_bytree"] = float(w["colsample_bytree"].value())
            elif key == "deepar":
                params["hidden_size"] = int(w["hidden_size"].value())
                params["num_layers"] = int(w["num_layers"].value())
                params["dropout"] = float(w["dropout"].value())
                params["epochs"] = int(w["epochs"].value())
                params["learning_rate"] = float(w["learning_rate"].value())
                params["batch_size"] = int(w["batch_size"].value())
                params["device"] = str(w["device"].currentText())
            else:
                continue

            models[key] = ModelConfig(name=key, params=params)

        if not models:
            raise ValueError("Please enable at least 1 model in the 'Models' tab.")

        return PipelineConfig(cleaning=cleaning, features=features, models=models)

    def _train(self) -> None:
        try:
            cfg = self._collect_config()
        except Exception as e:
            QMessageBox.warning(self, "Invalid config", str(e))
            return

        self._append_log(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training…")
        try:
            assert self.df is not None
            df_for_train = self._prepare_dataframe_for_training(self.df, cfg.features.time_col)
            result = run_training(df_for_train, cfg, log_cb=self._append_log)
        except Exception as e:
            QMessageBox.critical(self, "Training failed", f"{e}\n\n{traceback.format_exc()}")
            self._append_log("Training failed. See error dialog.")
            return

        self.last_result = result
        self._append_log("Training completed.")
        self._populate_visualization_controls()
        self._redraw_plot()
        self._populate_metrics_table()
        self._maybe_get_gpt_suggestions()

        QMessageBox.information(
            self,
            "Done",
            f"Training completed for {len(result.models)} model(s) and {len(result.targets)} target(s).",
        )

    def _prepare_dataframe_for_training(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        # If user selected "None", we synthesize a time column from row order.
        if time_col == "__ts__":
            out = df.copy()
            out["__ts__"] = pd.RangeIndex(start=0, stop=len(out), step=1)
            return out
        return df

    def _save_current_plot(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "No plot", "Train at least one model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            str(Path.home() / f"ts_plot_{self.last_result.run_id}.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.canvas.figure.savefig(path, dpi=200)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{e}\n\n{traceback.format_exc()}")
            return
        QMessageBox.information(self, "Saved", f"Saved plot to:\n{path}")

    def _clear_history_and_gpt(self) -> None:
        self.log.clear()
        self._reset_gpt_memory()

    def _set_openai_key_from_ui(self) -> None:
        key = self.openai_key_edit.text().strip()
        if not key:
            QMessageBox.information(self, "No key", "Paste an OpenAI API key first.")
            return
        # Basic sanity check; don't log or display the key.
        if len(key) < 20:
            QMessageBox.warning(self, "Key looks too short", "That key looks too short. Please paste the full key.")
            return
        os.environ["OPENAI_API_KEY"] = key
        self._append_log("OpenAI API key set for this app session (OPENAI_API_KEY).")
        QMessageBox.information(self, "Saved", "API key set for this app session.")

    def _on_openai_toggle(self, enabled: bool) -> None:
        for w in getattr(self, "_openai_key_widgets", []):
            w.setEnabled(bool(enabled))
        if not enabled:
            if hasattr(self, "log"):
                self._append_log("OpenAI advisor disabled.")

    def _reset_gpt_memory(self) -> None:
        self._gpt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior time-series ML engineer. "
                    "Given the user's current training settings and evaluation results, "
                    "suggest concrete improvements: which models to use/remove, "
                    "how to adjust hyperparameters, and which missing-value/outlier/scaling choices to try. "
                    "Be practical and concise. Provide 5-10 bullet points max."
                ),
            }
        ]

    def _maybe_get_gpt_suggestions(self) -> None:
        if self.last_result is None:
            return
        if not getattr(self, "enable_openai_chk", None) or not self.enable_openai_chk.isChecked():
            return
        if OpenAI is None:
            self._append_log(
                "GPT advisor unavailable: openai package not installed. Install from requirements.txt."
            )
            return

        # OpenAI SDK reads OPENAI_API_KEY from environment.
        try:
            client = OpenAI()
        except Exception as e:
            self._append_log(
                "GPT advisor not enabled. Set OPENAI_API_KEY (you can paste it in the UI).\n"
                f"Details: {e}"
            )
            return

        cfg = self.last_result.config
        metrics = self.last_result.metrics_by_model
        train_metrics = getattr(self.last_result, "train_metrics_by_model", {}) or {}

        selected_metrics = (
            (((cfg.get("features") or {}).get("eval_metrics")) or ["rmse"])
            if isinstance(cfg, dict)
            else ["rmse"]
        )
        primary_metric = str(selected_metrics[0]) if selected_metrics else "rmse"

        model_lines = []
        overfit_lines = []
        for m in self.last_result.models:
            label = MODEL_LABELS.get(m, m)
            mm = metrics.get(m, {})
            tm = train_metrics.get(m, {})
            model_lines.append(
                f"- {label}: "
                f"TRAIN({primary_metric}={tm.get(primary_metric)}) | "
                f"TEST({primary_metric}={mm.get(primary_metric)})"
            )
            try:
                # Overfitting detection MUST use the user-selected primary metric.
                tr = float(tm.get(primary_metric)) if tm.get(primary_metric) is not None else None
                te = float(mm.get(primary_metric)) if mm.get(primary_metric) is not None else None
                if tr and te and tr > 0:
                    overfit_lines.append(f"- {label}: test/train {primary_metric} ratio ≈ {te / tr:.3g}")
            except Exception:
                pass

        # Pre-select best model (lowest RMSE, then MAE) to constrain the advisor.
        def _score(m: str) -> tuple[float, float]:
            mm = metrics.get(m, {})
            primary = float(mm.get(primary_metric, float("inf")))
            rmse = float(mm.get("rmse", float("inf")))
            return (primary, rmse)

        best_model_key = sorted(self.last_result.models, key=_score)[0]
        best_model_label = MODEL_LABELS.get(best_model_key, best_model_key)
        overfit_threshold = 1.5
        best_overfit_ratio = None
        try:
            tm_best = train_metrics.get(best_model_key, {}) or {}
            mm_best = metrics.get(best_model_key, {}) or {}
            # MUST use the user-selected primary metric here too.
            tr = float(tm_best.get(primary_metric)) if tm_best.get(primary_metric) is not None else None
            te = float(mm_best.get(primary_metric)) if mm_best.get(primary_metric) is not None else None
            if tr and te and tr > 0:
                best_overfit_ratio = te / tr
        except Exception:
            best_overfit_ratio = None
        overfit_detected = bool(best_overfit_ratio is not None and best_overfit_ratio > overfit_threshold)

        ranges = {
            "MA": {"window": "1–5000"},
            "WMA": {"window": "1–5000"},
            "ARIMA": {"p": "0–10", "d": "0–3", "q": "0–10"},
            "PROPHET": {
                "changepoint_prior_scale": "0.001–2.0",
                "seasonality_prior_scale": "0.01–20.0",
                "n_changepoints": "0–100",
                "seasonality_mode": "additive|multiplicative",
            },
            "XGB": {
                "max_depth": "1–20",
                "learning_rate": "0.0001–1.0",
                "n_estimators": "10–5000",
                "subsample": "0.2–1.0",
                "colsample_bytree": "0.2–1.0",
            },
            "DeepAR": {
                "hidden_size": "8–512",
                "num_layers": "1–4",
                "dropout": "0.0–0.8",
                "epochs": "1–200",
                "learning_rate": "1e-5–1e-1",
                "batch_size": "8–1024",
            },
        }

        allowed_cleaning = {
            "missing_method": [code for _, code in MISSING_METHOD_ITEMS],
            "outlier_method": [code for _, code in OUTLIER_METHOD_ITEMS],
            "scaler": [code for _, code in SCALER_ITEMS],
            "target_transform": [code for _, code in TARGET_TRANSFORM_ITEMS],
            "clip_q_low": "0.00–0.49",
            "clip_q_high": "0.51–1.00",
        }

        priority_rule = (
            "FIRST STEP: Decide whether overfitting exists from TRAIN vs TEST. "
            "If yes, you must ONLY give overfitting-mitigation advice. "
            "If no, you must ONLY give performance-improvement advice."
        )
        branch_rule = (
            "Overfitting branch: detected=True, so output ONLY overfitting-mitigation changes "
            "(regularization / simplifying the chosen model / reducing capacity). "
            "Do NOT include general performance tuning yet."
            if overfit_detected
            else "Performance branch: detected=False, so output ONLY performance-improvement changes on TEST. "
            "Do NOT include overfitting-mitigation changes."
        )

        user_msg = (
            "You must produce ONLY actionable settings that exist in the UI.\n"
            "- Do NOT mention methods we do not provide (e.g., z-score outlier removal, isolation forest, etc.).\n"
            "- If you suggest a cleaning change, the new value MUST be one of the allowed options.\n"
            "- Hyperparameter changes MUST stay within the provided ranges.\n"
            "- You must choose exactly ONE best model among the models trained in this run (no new models).\n"
            f"- {priority_rule}\n"
            f"- {branch_rule}\n\n"
            f"Best-by-metrics hint (computed): {best_model_label} (key={best_model_key}).\n\n"
            f"OverfittingDetected (heuristic for best model): {overfit_detected}. "
            f"best_model_test/train_ratio={best_overfit_ratio} (threshold={overfit_threshold}).\n\n"
            f"Targets: {self.last_result.targets}\n"
            f"Models trained (keys): {self.last_result.models}\n"
            f"Models trained (labels): {[MODEL_LABELS.get(m, m) for m in self.last_result.models]}\n\n"
            "Current settings (JSON):\n"
            f"{json.dumps(cfg, indent=2, ensure_ascii=False)}\n\n"
            "Aggregated metrics (lower is better). TRAIN is in-sample; TEST is held-out:\n"
            + "\n".join(model_lines)
            + "\n\n"
            f"Overfitting signal (test/train {primary_metric} ratio; >{overfit_threshold} is suspicious):\n"
            + ("\n".join(overfit_lines) if overfit_lines else "- (not available)")
            + "\n\n"
            "Allowed cleaning options (MUST use these exact codes):\n"
            f"{json.dumps(allowed_cleaning, indent=2, ensure_ascii=False)}\n\n"
            "Allowed hyperparameter ranges (UI constraints):\n"
            f"{json.dumps(ranges, indent=2, ensure_ascii=False)}\n\n"
            "Required output format:\n"
            "1) Best model: <MODEL_KEY> - <MODEL_LABEL>\n"
            "2) Changes (each must be explicit old->new):\n"
            f"   - Your branch for this run is: {'OVERFITTING' if overfit_detected else 'PERFORMANCE'}.\n"
            "- Cleaning: <setting>: <old> -> <new>\n"
            "- Hyperparameters: <MODEL_KEY>.<param>: <old> -> <new>\n"
            "- If you suggest any XGB or DeepAR hyperparameter changes, you MUST explicitly address learning_rate "
            "(either change it with old->new or state it stays the same).\n"
            "- Data split/horizon/lags: <setting>: <old> -> <new>\n"
            "3) Overfitting check: <yes/no> and why (reference the train/test numbers).\n"
            "4) One-sentence rationale.\n"
        )

        self._append_log("\n--- GPT advisor: requesting suggestions… ---\n")
        try:
            messages = list(self._gpt_messages) + [{"role": "user", "content": user_msg}]
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            self._append_log(f"GPT request failed: {e}\n\n{traceback.format_exc()}")
            return

        # Validate applicability; retry once if GPT suggests unsupported options.
        issues = self._validate_gpt_advice(text, allowed_cleaning=allowed_cleaning)
        if issues:
            self._append_log(
                "GPT advice had unsupported suggestions; retrying with constraints.\n"
                + "\n".join([f"- {x}" for x in issues])
                + "\n"
            )
            try:
                repair_msg = (
                    "Your previous answer contained unsupported options.\n"
                    "Fix it and output again using ONLY allowed UI options.\n\n"
                    "Unsupported items detected:\n"
                    + "\n".join([f"- {x}" for x in issues])
                    + "\n\n"
                    "Remember: for cleaning changes, the NEW value must be one of the allowed codes:\n"
                    f"{json.dumps(allowed_cleaning, indent=2, ensure_ascii=False)}\n"
                )
                messages2 = list(self._gpt_messages) + [{"role": "user", "content": user_msg}, {"role": "user", "content": repair_msg}]
                resp2 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages2,
                    temperature=0.1,
                )
                text2 = resp2.choices[0].message.content or ""
                issues2 = self._validate_gpt_advice(text2, allowed_cleaning=allowed_cleaning)
                if not issues2:
                    text = text2
                else:
                    self._append_log(
                        "GPT retry still contained unsupported suggestions; showing best-effort output.\n"
                        + "\n".join([f"- {x}" for x in issues2])
                    )
                    text = text2
            except Exception as e:
                self._append_log(f"GPT retry failed: {e}\n\n{traceback.format_exc()}")

        self._gpt_messages.append({"role": "user", "content": user_msg})
        self._gpt_messages.append({"role": "assistant", "content": text})
        self._append_log(text.strip() + "\n")

    def _validate_gpt_advice(self, text: str, allowed_cleaning: Dict[str, Any]) -> List[str]:
        """
        Best-effort validator for the constrained output format.
        Returns a list of issues (empty => looks applicable).
        """
        issues: List[str] = []
        if not text.strip():
            return ["empty response"]

        # Validate "Best model" key if present
        m_best = re.search(r"^\s*1\)\s*Best model:\s*([A-Za-z0-9_]+)\s*-", text, flags=re.MULTILINE)
        if m_best:
            key = m_best.group(1).strip()
            if self.last_result is not None and key not in self.last_result.models:
                issues.append(f"Best model '{key}' is not one of the trained models.")

        # Validate cleaning new values are allowed codes
        # Expected line: "- Cleaning: outlier_method: old -> new"
        for m in re.finditer(r"^-+\s*Cleaning:\s*([A-Za-z0-9_]+)\s*:\s*(.*?)\s*->\s*(.*?)\s*$", text, flags=re.MULTILINE):
            setting = m.group(1).strip()
            new_val = m.group(3).strip()
            if setting in {"missing_method", "outlier_method", "scaler", "target_transform"}:
                allowed = set(allowed_cleaning.get(setting, []))
                if new_val not in allowed:
                    issues.append(f"Cleaning.{setting} new value '{new_val}' not in allowed {sorted(allowed)}")

        # Quick heuristic: reject common unsupported outlier suggestions
        lower = text.lower()
        if "z-score" in lower or "zscore" in lower:
            if "outlier" in lower:
                issues.append("Mentions z-score outlier handling (not available).")
        if "isolation forest" in lower or "lof" in lower:
            issues.append("Mentions outlier methods not available in UI.")

        return issues

    def _populate_visualization_controls(self) -> None:
        self.target_view_combo.blockSignals(True)
        self.target_view_combo.clear()
        if self.last_result is not None:
            self.target_view_combo.addItems(self.last_result.targets)
        self.target_view_combo.blockSignals(False)

        # model line toggles
        while self.model_lines_layout.count():
            item = self.model_lines_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if self.last_result is None:
            return

        self._line_toggles: Dict[str, QCheckBox] = {}
        for model_name in self.last_result.models:
            chk = QCheckBox(MODEL_LABELS.get(model_name, model_name))
            chk.setChecked(True)
            chk.stateChanged.connect(self._redraw_plot)
            self.model_lines_layout.addWidget(chk)
            self._line_toggles[model_name] = chk

        self.model_lines_layout.addStretch(1)

    def _redraw_plot(self, *args: object) -> None:
        self.canvas.ax.clear()
        if self.last_result is None:
            self.canvas.draw()
            return

        target = self.target_view_combo.currentText()
        if not target:
            self.canvas.draw()
            return

        series = self.last_result.per_target[target]
        t = series["t"]
        y_true = series["y_true"]

        self.canvas.ax.plot(t, y_true, label="Actual", linewidth=2.2, alpha=0.9)

        for model_name, pred in series["preds"].items():
            if hasattr(self, "_line_toggles"):
                if model_name in self._line_toggles and not self._line_toggles[model_name].isChecked():
                    continue
            self.canvas.ax.plot(t, pred, label=MODEL_LABELS.get(model_name, model_name), linewidth=1.8, alpha=0.9)

        self.canvas.ax.set_title(f"Forecast on test window (target: {target})")
        self.canvas.ax.set_xlabel("Time")
        self.canvas.ax.set_ylabel("Value")
        self.canvas.ax.grid(True, alpha=0.25)
        self.canvas.ax.legend()
        self.canvas.draw()

    def _populate_metrics_table(self) -> None:
        if self.last_result is None:
            self.metrics_table.clear()
            return

        # Flatten metrics: rows = model, cols = selected metrics (from config)
        cfg = self.last_result.config or {}
        eval_metrics = (
            (((cfg.get("features") or {}).get("eval_metrics")) or ["rmse", "mae", "mape_pct"])
            if isinstance(cfg, dict)
            else ["rmse", "mae", "mape_pct"]
        )
        metric_labels = {code: label for (label, code) in EVAL_METRIC_ITEMS}

        models = list(self.last_result.models)
        cols = [metric_labels.get(c, c) for c in eval_metrics]
        self.metrics_table.setRowCount(len(models))
        self.metrics_table.setColumnCount(len(cols))
        self.metrics_table.setHorizontalHeaderLabels(cols)
        self.metrics_table.setVerticalHeaderLabels([MODEL_LABELS.get(m, m) for m in models])

        for r, model_name in enumerate(models):
            m = self.last_result.metrics_by_model.get(model_name, {})
            for cidx, code in enumerate(eval_metrics):
                v = m.get(code)
                if isinstance(v, (int, float)):
                    self.metrics_table.setItem(r, cidx, QTableWidgetItem(f"{float(v):.6g}"))
                else:
                    self.metrics_table.setItem(r, cidx, QTableWidgetItem(str(v)))
        self.metrics_table.resizeColumnsToContents()

    def _save_last_result(self) -> None:
        if self.last_result is None:
            QMessageBox.warning(self, "No result", "Train at least one model first.")
            return

        default_name = "my_run"
        name, ok = QInputDialog.getText(
            self,
            "Save result",
            "Record name (will be shown in Saved Results):",
            QLineEdit.Normal,
            default_name,
        )
        if not ok:
            return
        display_name = name.strip() or default_name
        slug = _slugify_name(display_name) or "run"
        run_id = f"{_now_run_id()}__{slug}"

        # Avoid collisions if user saves twice in the same second
        suffix = 1
        while self.store.run_dir(run_id).exists():
            suffix += 1
            run_id = f"{_now_run_id()}__{slug}_{suffix}"

        try:
            run_path = self.store.save_run(run_id=run_id, result=self.last_result, display_name=display_name)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{e}\n\n{traceback.format_exc()}")
            return

        self._refresh_saved_runs()
        QMessageBox.information(self, "Saved", f"Saved run to:\n{run_path}")

    # -------------------------
    # Tab 4: Saved Results
    # -------------------------
    def _build_saved_tab(self) -> None:
        layout = QHBoxLayout()
        self.tab_saved.setLayout(layout)

        left = QWidget()
        left_layout = QVBoxLayout()
        left.setLayout(left_layout)
        layout.addWidget(left, 1)

        left_layout.addWidget(QLabel("Saved runs:"))
        self.runs_list = QListWidget()
        self.runs_list.currentItemChanged.connect(self._show_run_details)
        left_layout.addWidget(self.runs_list, 1)

        btns = QHBoxLayout()
        left_layout.addLayout(btns)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_saved_runs)
        btns.addWidget(btn_refresh)

        btn_export = QPushButton("Export selected…")
        btn_export.clicked.connect(self._export_selected_run)
        btns.addWidget(btn_export)

        btn_delete = QPushButton("Delete selected")
        btn_delete.clicked.connect(self._delete_selected_run)
        btns.addWidget(btn_delete)

        right = QWidget()
        right_layout = QVBoxLayout()
        right.setLayout(right_layout)
        layout.addWidget(right, 2)

        right_layout.addWidget(QLabel("Run details:"))
        self.run_details = QPlainTextEdit()
        self.run_details.setReadOnly(True)
        right_layout.addWidget(self.run_details, 2)

        self.saved_canvas = MplCanvas()
        right_layout.addWidget(QLabel("Saved plot preview:"))
        right_layout.addWidget(self.saved_canvas, 3)

    def _refresh_saved_runs(self) -> None:
        self.runs_list.clear()
        runs = self.store.list_runs()
        for run in runs:
            shown = run.display_name.strip() or run.run_id
            it = QListWidgetItem(f"{shown}  [{_human_ts(run.run_id)}]")
            it.setData(Qt.UserRole, run.run_id)
            self.runs_list.addItem(it)

    def _selected_run_id(self) -> Optional[str]:
        it = self.runs_list.currentItem()
        if it is None:
            return None
        return it.data(Qt.UserRole)

    def _show_run_details(self, *args: object) -> None:
        run_id = self._selected_run_id()
        self.run_details.clear()
        self.saved_canvas.ax.clear()
        self.saved_canvas.draw()
        if not run_id:
            return

        try:
            info = self.store.load_run_info(run_id)
        except Exception as e:
            self.run_details.setPlainText(f"Failed to load run:\n{e}\n\n{traceback.format_exc()}")
            return

        self.run_details.setPlainText(json.dumps(info, indent=2, ensure_ascii=False))

        # Try load preview plot if exists
        plot_path = self.store.run_dir(run_id) / "plot.png"
        if plot_path.exists():
            import matplotlib.image as mpimg

            img = mpimg.imread(str(plot_path))
            self.saved_canvas.ax.imshow(img)
            self.saved_canvas.ax.axis("off")
            self.saved_canvas.draw()

    def _delete_selected_run(self) -> None:
        run_id = self._selected_run_id()
        if not run_id:
            QMessageBox.information(self, "No selection", "Select a run first.")
            return
        if QMessageBox.question(self, "Confirm", f"Delete run {run_id}?") != QMessageBox.Yes:
            return
        try:
            self.store.delete_run(run_id)
        except Exception as e:
            QMessageBox.critical(self, "Delete failed", f"{e}\n\n{traceback.format_exc()}")
            return
        self._refresh_saved_runs()
        self.run_details.clear()
        QMessageBox.information(self, "Deleted", f"Deleted run {run_id}.")

    def _export_selected_run(self) -> None:
        run_id = self._selected_run_id()
        if not run_id:
            QMessageBox.information(self, "No selection", "Select a run first.")
            return
        target_dir = QFileDialog.getExistingDirectory(self, "Choose export folder", str(Path.home()))
        if not target_dir:
            return
        try:
            exported = self.store.export_run(run_id, Path(target_dir))
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"{e}\n\n{traceback.format_exc()}")
            return
        QMessageBox.information(self, "Exported", f"Exported to:\n{exported}")


def main() -> int:
    # Prophet may emit a noisy error at import time if optional plotly isn't installed:
    # "Importing plotly failed. Interactive plots will not work."
    # We don't use Prophet's plotly integration in this app, so silence it.
    logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)
    logging.getLogger("prophet.plot").propagate = False

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    # Larger, more readable default font and components.
    base_font = QFont()
    base_font.setPointSize(12)
    app.setFont(base_font)
    # qt-material themes can make checkbox indicators hard to see depending on palette;
    # force a readable checkbox style.
    app.setStyleSheet(
        app.styleSheet()
        + """
QWidget { font-size: 12pt; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  min-height: 34px;
  padding: 6px 10px;
}
QPushButton {
  min-height: 38px;
  padding: 8px 14px;
  font-weight: 600;
}
QTabBar::tab {
  min-height: 34px;
  min-width: 180px;
  padding: 10px 14px;
  font-weight: 600;
}
QGroupBox { font-weight: 700; }
QLabel { font-size: 12pt; }

QCheckBox { color: #1f1f1f; }
QCheckBox::indicator {
  width: 18px;
  height: 18px;
  border: 1px solid #5a5a5a;
  border-radius: 3px;
  background: #ffffff;
}
QCheckBox::indicator:checked {
  background: #2979ff;
  border: 1px solid #2979ff;
}
"""
    )
    w = MainWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

