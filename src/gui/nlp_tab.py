"""NLP Tab: datasets, classifiers, custom dictionaries editor, and visualization."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
)

from src.gui.plot_canvas import PlotCanvas
from src.gui.workers import WorkerThread
from src.nlp.classification import CLASSIFIERS, NLPParams, compare_classifiers, evaluate, study_max_features, study_ngram
from src.nlp.datasets import DATASET_LABELS, TextDataset, load_dataset
from src.nlp.dictionaries import DEFAULT_DICTIONARIES
from src.system_info import get_machine_info


class NLPTab(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset: TextDataset | None = None
        self.dictionaries = {name: list(words) for name, words in DEFAULT_DICTIONARIES.items()}
        self._worker: WorkerThread | None = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(15)

        # Control Panel Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(380)

        scroll_content = QWidget()
        left = QVBoxLayout(scroll_content)
        left.setSpacing(8)
        left.setContentsMargins(0, 0, 5, 0)

        # Group 1: Dataset Loader
        group_dataset = QGroupBox("Incarcare Date")
        group_dataset_layout = QVBoxLayout(group_dataset)
        self.dataset_combo = QComboBox()
        for key, label in DATASET_LABELS.items():
            self.dataset_combo.addItem(label, key)
        group_dataset_layout.addWidget(QLabel("Selecteaza Set de date:"))
        group_dataset_layout.addWidget(self.dataset_combo)
        self.btn_load = QPushButton("Incarca Set Date")
        self.btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        group_dataset_layout.addWidget(self.btn_load)
        
        self.info_label = QLabel("Niciun set de date incarcat.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #475569; font-weight: 500;")
        group_dataset_layout.addWidget(self.info_label)
        left.addWidget(group_dataset)

        # Group 2: Pipeline Parameters
        group_params = QGroupBox("Parametri Pipeline")
        form = QFormLayout(group_params)
        self.clf_combo = QComboBox()
        self.clf_combo.addItems(list(CLASSIFIERS.keys()))
        form.addRow("Clasificator:", self.clf_combo)

        self.ngram1 = QSpinBox()
        self.ngram1.setRange(1, 3)
        self.ngram1.setValue(1)
        self.ngram2 = QSpinBox()
        self.ngram2.setRange(1, 3)
        self.ngram2.setValue(2)
        ng_row = QHBoxLayout()
        ng_row.addWidget(self.ngram1)
        ng_row.addWidget(QLabel("to"))
        ng_row.addWidget(self.ngram2)
        form.addRow("ngram_range:", ng_row)

        self.max_feat = QSpinBox()
        self.max_feat.setRange(0, 200000)
        self.max_feat.setSpecialValueText("toate (all)")
        self.max_feat.setValue(10000)
        form.addRow("max_features:", self.max_feat)

        # Dictionary Feature Union toggle checkbox
        self.chk_use_dict = QCheckBox("Utilizeaza Dictionare (Lexicoane)")
        self.chk_use_dict.setChecked(False)
        form.addRow(self.chk_use_dict)
        left.addWidget(group_params)

        # Group 3: Dictionaries Real-time Editor
        group_dict = QGroupBox("Editor Dictionare NLP")
        group_dict_layout = QVBoxLayout(group_dict)

        self.dict_edit_combo = QComboBox()
        self.dict_edit_combo.addItems(list(self.dictionaries.keys()))
        group_dict_layout.addWidget(QLabel("Alege dictionarul de editat:"))
        group_dict_layout.addWidget(self.dict_edit_combo)

        self.dict_edit_text = QTextEdit()
        self.dict_edit_text.setMaximumHeight(100)
        self.dict_edit_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        group_dict_layout.addWidget(QLabel("Cuvinte (unul pe linie):"))
        group_dict_layout.addWidget(self.dict_edit_text)

        self.btn_save_dict = QPushButton("Salveaza Cuvinte Dictionar")
        self.btn_save_dict.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        group_dict_layout.addWidget(self.btn_save_dict)
        left.addWidget(group_dict)

        # Group 4: Training & Evaluation Actions
        group_actions = QGroupBox("Rulare Experimente")
        actions_layout = QVBoxLayout(group_actions)

        row1 = QHBoxLayout()
        self.btn_train = QPushButton("Antreneaza & Evalueaza")
        self.btn_compare = QPushButton("Compara Clasificatori")
        self.btn_train.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_compare.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row1.addWidget(self.btn_train)
        row1.addWidget(self.btn_compare)
        actions_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_ngram = QPushButton("Studiu ngram_range")
        self.btn_features = QPushButton("Studiu max_features")
        self.btn_ngram.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_features.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row2.addWidget(self.btn_ngram)
        row2.addWidget(self.btn_features)
        actions_layout.addLayout(row2)
        left.addWidget(group_actions)

        left.addStretch()

        scroll.setWidget(scroll_content)
        root.addWidget(scroll, 0)

        # Right side: canvas + report below (full width)
        right_side = QVBoxLayout()
        right_side.setSpacing(6)
        right_side.setContentsMargins(0, 0, 0, 0)

        self.canvas = PlotCanvas()
        right_side.addWidget(self.canvas, 1)

        # Hidden report storage (backwards compatibility)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setVisible(False)

        # Visible console (styled QLabel inside scroll area) — like TSP
        self.console_label = QLabel()
        self.console_label.setWordWrap(True)
        self.console_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.console_label.setTextFormat(Qt.PlainText)
        self.console_label.setStyleSheet(
            "QLabel { font-family: Consolas, monospace; font-size: 11px;"
            " padding: 8px 10px; color: #ffffff; }"
        )
        self.console_label.setText("Niciun raport disponibil.")

        console_scroll = QScrollArea()
        console_scroll.setWidget(self.console_label)
        console_scroll.setWidgetResizable(True)
        console_scroll.setFrameShape(QScrollArea.NoFrame)
        console_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Increase console height to avoid showing scrollbar by default
        console_scroll.setMinimumHeight(160)
        console_scroll.setMaximumHeight(360)
        console_scroll.setStyleSheet(
            "QScrollArea { background: #1e293b; border-top: 2px solid #334155; }"
        )
        right_side.addWidget(console_scroll, 0)

        root.addLayout(right_side, 1)

        # Event connections
        self.btn_load.clicked.connect(self._load_dataset)
        self.btn_train.clicked.connect(self._train)
        self.btn_compare.clicked.connect(self._compare)
        self.btn_ngram.clicked.connect(self._study_ngram)
        self.btn_features.clicked.connect(self._study_features)
        self.dict_edit_combo.currentTextChanged.connect(self._load_dict_words)
        self.btn_save_dict.clicked.connect(self._save_dict_words)

        # Load initial dictionary words for the first dictionary
        self._load_dict_words(self.dict_edit_combo.currentText())

    def _set_report_text(self, text: str):
        """Update hidden report storage and visible console label."""
        try:
            self.report.setPlainText(text)
        except Exception:
            pass
        self.console_label.setText(text)

    def _load_dict_words(self, dict_name: str):
        if dict_name in self.dictionaries:
            self.dict_edit_text.setPlainText("\n".join(self.dictionaries[dict_name]))

    def _save_dict_words(self):
        dict_name = self.dict_edit_combo.currentText()
        if dict_name in self.dictionaries:
            text = self.dict_edit_text.toPlainText()
            words = [w.strip() for w in text.splitlines() if w.strip()]
            self.dictionaries[dict_name] = words
            self._set_report_text(
                f"Modificari salvate! Dictionarul '{dict_name}' are acum {len(words)} cuvinte."
            )

    def _params(self) -> NLPParams:
        mf = self.max_feat.value()
        return NLPParams(
            classifier=self.clf_combo.currentText(),
            ngram_range=(self.ngram1.value(), self.ngram2.value()),
            max_features=None if mf == 0 else mf,
            use_dictionaries=self.chk_use_dict.isChecked(),
        )

    def _set_busy(self, busy: bool):
        for btn in (
            self.btn_load,
            self.btn_train,
            self.btn_compare,
            self.btn_ngram,
            self.btn_features,
            self.btn_save_dict,
        ):
            btn.setEnabled(not busy)

    def _run_async(self, fn, on_ok):
        if self._worker and self._worker.isRunning():
            return
        self._set_busy(True)
        self._worker = WorkerThread(fn)
        self._worker.finished_ok.connect(on_ok)
        self._worker.failed.connect(lambda m: self._set_report_text(f"Eroare: {m}"))
        self._worker.finished.connect(lambda: self._set_busy(False))
        self._worker.start()

    def _load_dataset(self):
        key = self.dataset_combo.currentData()
        name = self.dataset_combo.currentText()
        self.info_label.setText(f"Se descarcă / se încarcă:\n{name}...\nVă rugăm așteptați (poate dura)...")
        self._set_report_text(f"Descărcare și procesare set de date '{name}' în curs de desfășurare. Vă rugăm așteptați...")

        def job():
            return load_dataset(key)

        def done(ds: TextDataset):
            self.dataset = ds
            self.info_label.setText(
                f"Set date incarcat: {ds.name}\n"
                f"Train: {ds.n_train} | Test: {ds.n_test}\n"
                f"Clase: {', '.join(ds.target_names)}"
            )
            self._set_report_text(f"Succes! Datele pentru {ds.name} au fost incarcate.")

        self._run_async(job, done)

    def _train(self):
        if not self.dataset:
            self._set_report_text("Incarcati un set de date mai intai.")
            return
        ds, params = self.dataset, self._params()
        dicts = self.dictionaries
        self._set_report_text(f"Antrenare în curs pentru clasificatorul '{params.classifier}'... Vă rugăm așteptați...")

        def job():
            return evaluate(ds, params, dicts)

        def done(res):
            self._set_report_text(
                f"Acuratete: {res.accuracy:.4f}\n"
                f"Timp antrenare: {res.train_time_s:.2f} s\n"
                f"Foloseste Dictionare: {params.use_dictionaries}\n\n"
                f"{res.report}"
            )
            self._plot_confusion(
                res.confusion, ds.target_names, f"{params.classifier} — acc {res.accuracy:.3f}"
            )

        self._run_async(job, done)

    def _compare(self):
        if not self.dataset:
            self._set_report_text("Incarcati un set de date mai intai.")
            return
        ds = self.dataset
        base = self._params()
        dicts = self.dictionaries
        self._set_report_text("Comparare clasificatori în curs de desfășurare... Vă rugăm așteptați (poate dura câteva zeci de secunde)...")

        def job():
            return compare_classifiers(ds, base, dicts)

        def done(results):
            names = [r.params.classifier for r in results]
            accs = [r.accuracy for r in results]
            self._plot_bars(names, accs, "Comparatie Clasificatori", "Clasificator")
            lines = [
                f"{r.params.classifier}: acc={r.accuracy:.4f}, timp={r.train_time_s:.2f}s"
                for r in results
            ]
            best = max(results, key=lambda r: r.accuracy)
            lines.append(f"\nCel mai bun: {best.params.classifier}")
            lines.append(f"Foloseste Dictionare: {base.use_dictionaries}")
            self._set_report_text("\n".join(lines))

        self._run_async(job, done)

    def _study_ngram(self):
        if not self.dataset:
            self._set_report_text("Incarcati un set de date mai intai.")
            return
        ds = self.dataset
        base = self._params()
        dicts = self.dictionaries
        self._set_report_text(f"Studiu ngram_range în curs pentru {base.classifier}... Vă rugăm așteptați...")

        def job():
            return study_ngram(ds, base=base, dictionaries=dicts)

        def done(results):
            labels = [str(r.params.ngram_range) for r in results]
            accs = [r.accuracy for r in results]
            self._plot_bars(labels, accs, f"Studiu ngram_range ({base.classifier})", "ngram_range")
            self._set_report_text(
                f"Studiu ngram_range (Foloseste Dictionare: {base.use_dictionaries}):\n"
                + "\n".join(f"ngram={r.params.ngram_range}: {r.accuracy:.4f}" for r in results)
            )

        self._run_async(job, done)

    def _study_features(self):
        if not self.dataset:
            self._set_report_text("Incarcati un set de date mai intai.")
            return
        ds = self.dataset
        base = self._params()
        dicts = self.dictionaries
        self._set_report_text(f"Studiu max_features în curs pentru {base.classifier}... Vă rugăm așteptați...")

        def job():
            return study_max_features(ds, base=base, dictionaries=dicts)

        def done(results):
            labels = [str(r.params.max_features or "all") for r in results]
            accs = [r.accuracy for r in results]
            self._plot_bars(labels, accs, f"Studiu max_features ({base.classifier})", "max_features")
            self._set_report_text(
                f"Studiu max_features (Foloseste Dictionare: {base.use_dictionaries}):\n"
                + "\n".join(f"max_features={r.params.max_features}: {r.accuracy:.4f}" for r in results)
            )

        self._run_async(job, done)

    def _plot_bars(self, labels: list[str], values: list[float], title: str, xlabel: str):
        self.canvas.clear()
        ax = self.canvas.ax
        ax.bar(labels, values, color="steelblue", edgecolor="black", width=0.4)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Acuratete")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, linestyle="--", alpha=0.3)
        self.canvas.refresh()

    def _plot_confusion(self, cm, labels: list[str], title: str):
        self.canvas.clear()
        ax = self.canvas.ax
        im = ax.imshow(cm, cmap="Blues")
        self.canvas.fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prezis")
        ax.set_ylabel("Real")
        ax.set_title(title)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9, fontweight="bold")
        self.canvas.refresh()
