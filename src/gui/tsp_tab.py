"""TSP Tab: algorithms, parameters, visualization, benchmarking, and matrix saving."""

from __future__ import annotations

import math
import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QCheckBox,
    QTextEdit,
    QTextBrowser,
    QLineEdit,
    QTabWidget,
    QScrollArea,
    QSizePolicy,
)

from src.gui.plot_canvas import PlotCanvas
from src.gui.workers import WorkerThread
from src.system_info import get_machine_info
from src.tsp.algorithms import (
    BKTParams,
    GAParams,
    HCParams,
    NNParams,
    SAParams,
    SOLVERS,
    TSPResult,
)
from src.tsp.benchmarks import run_tsp_benchmark
from src.tsp.utils import (
    format_tour,
    generate_cities,
    load_matrix_file,
    matrix_from_coords,
    save_matrix_file,
    random_matrix,
    tour_cost,
)


class TSPTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cities: list[tuple[float, float]] = []
        self.dist_matrix: list[list[float]] = []
        self._worker: WorkerThread | None = None
        self._build_ui()
        self._generate_cities()

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
        controls = QVBoxLayout(scroll_content)
        controls.setSpacing(10)
        controls.setContentsMargins(0, 0, 5, 0)

        # 1. Data Source & Generation Group
        group_data = QGroupBox("Generare / Încărcare Date")
        group_data_layout = QVBoxLayout(group_data)
        form_data = QFormLayout()

        self.spin_n = QSpinBox()
        self.spin_n.setRange(3, 150)
        self.spin_n.setValue(10)
        form_data.addRow("Număr orașe (N):", self.spin_n)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        form_data.addRow("Seed aleator:", self.seed_spin)
        group_data_layout.addLayout(form_data)

        # Generation buttons
        btn_gen_row = QHBoxLayout()
        self.btn_gen = QPushButton("Gen. Orașe (Euclidian)")
        self.btn_gen_non_eucl = QPushButton("Gen. Non-Euclidian")
        self.btn_gen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_gen_non_eucl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_gen_row.addWidget(self.btn_gen)
        btn_gen_row.addWidget(self.btn_gen_non_eucl)
        group_data_layout.addLayout(btn_gen_row)

        # File I/O buttons
        btn_file_row = QHBoxLayout()
        self.btn_load = QPushButton("Încarcă Matrică")
        self.btn_save = QPushButton("Salvează Matrică")
        self.btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_save.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_file_row.addWidget(self.btn_load)
        btn_file_row.addWidget(self.btn_save)
        group_data_layout.addLayout(btn_file_row)

        controls.addWidget(group_data)

        # 2. Algorithm & Parameters
        group_algo = QGroupBox("Configurare Algoritm")
        group_algo_layout = QVBoxLayout(group_algo)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(SOLVERS.keys()))
        group_algo_layout.addWidget(QLabel("Algoritm selectat:"))
        group_algo_layout.addWidget(self.algo_combo)

        self.params_box = QGroupBox("Parametri Personalizați")
        self.params_layout = QFormLayout(self.params_box)
        group_algo_layout.addWidget(self.params_box)
        self._param_widgets: dict[str, QWidget] = {}
        self.algo_combo.currentTextChanged.connect(self._rebuild_params)
        self._rebuild_params(self.algo_combo.currentText())

        controls.addWidget(group_algo)

        # 3. Execution & Complex Benchmarking
        group_run = QGroupBox("Rulare și Experimente")
        group_run_layout = QVBoxLayout(group_run)

        run_btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Rulare Algoritm")
        self.btn_compare = QPushButton("Compară Algoritmi")
        self.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_compare.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        run_btn_row.addWidget(self.btn_run)
        run_btn_row.addWidget(self.btn_compare)
        group_run_layout.addLayout(run_btn_row)

        # Benchmark Settings
        bench_form = QFormLayout()
        self.line_sizes = QLineEdit("10, 20, 50, 75, 100")
        bench_form.addRow("Dimensiuni N:", self.line_sizes)

        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 10)
        self.spin_repeats.setValue(3)
        bench_form.addRow("Repetări per N:", self.spin_repeats)
        group_run_layout.addLayout(bench_form)

        self.btn_bench = QPushButton("Rulare Benchmark Complexe")
        self.btn_bench.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        group_run_layout.addWidget(self.btn_bench)
        controls.addWidget(group_run)

        # Progress bar + status label (centered)
        self.progress_label = QLabel("Gata.")
        self.progress_label.setStyleSheet("color: #475569; font-size: 11px;")
        self.progress_label.setAlignment(Qt.AlignHCenter)
        controls.addWidget(self.progress_label, 0, Qt.AlignHCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setFixedWidth(320)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background: #e2e8f0;
                text-align: center;
                font-size: 11px;
                font-weight: bold;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #6366f1);
                border-radius: 8px;
            }
        """)
        controls.addWidget(self.progress_bar, 0, Qt.AlignHCenter)

        controls.addStretch()
        
        scroll.setWidget(scroll_content)
        root.addWidget(scroll, 0)
        
        # Right Side Layout (Graphics + Display at bottom)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # Visualization Tabs
        self.viz_tabs = QTabWidget()
        
        self.canvas_map = PlotCanvas()
        self.canvas_conv = PlotCanvas()
        self.canvas_comp = PlotCanvas()
        self.canvas_bench = PlotCanvas()

        self.viz_tabs.addTab(self.canvas_map, "Harta si Rute")
        self.viz_tabs.addTab(self.canvas_conv, "Convergenta")
        self.viz_tabs.addTab(self.canvas_comp, "Comparare")
        self.viz_tabs.addTab(self.canvas_bench, "Benchmark")
        
        right_layout.addWidget(self.viz_tabs, 1)

        # Hidden widgets kept for backward compat with internal calls
        self.result_text = QTextEdit()
        self.result_text.setVisible(False)
        self.result_display = self.result_text  # redirect all setHtml to hidden widget

        # Console — fills all remaining height below graphs
        self.console_label = QLabel()
        self.console_label.setWordWrap(True)
        self.console_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.console_label.setTextFormat(Qt.PlainText)
        self.console_label.setStyleSheet(
            "QLabel { font-family: Consolas, monospace; font-size: 12px;"
            " padding: 8px 10px; color: #ffffff; line-height: 1.6; }"
        )
        self.console_label.setText("Asteapta rulare algoritm...")

        console_scroll = QScrollArea()
        console_scroll.setWidget(self.console_label)
        console_scroll.setWidgetResizable(True)
        console_scroll.setFrameShape(QScrollArea.NoFrame)
        console_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        console_scroll.setStyleSheet(
            "QScrollArea { background: #1e293b; border-top: 2px solid #334155; }"
        )
        right_layout.addWidget(console_scroll, 1)  # stretch=1, fills remaining space

        root.addLayout(right_layout, 1)


        # Event connections
        self.btn_gen.clicked.connect(self._generate_cities)
        self.btn_gen_non_eucl.clicked.connect(self._generate_non_euclidian)
        self.btn_load.clicked.connect(self._load_file)
        self.btn_save.clicked.connect(self._save_file)
        self.btn_run.clicked.connect(self._run_single)
        self.btn_compare.clicked.connect(self._run_compare)
        self.btn_bench.clicked.connect(self._run_benchmark)

    def _clear_params(self):
        while self.params_layout.rowCount():
            self.params_layout.removeRow(0)
        self._param_widgets.clear()

    def _rebuild_params(self, algo: str):
        self._clear_params()
        if algo == "BKT":
            self._add_combo("mode", ["exhaustiv", "prima", "y_solutii", "timp"], 0)
            self._add_spin("y_solutions", 1, 1000, 5)
            self._add_float("time_limit_s", 1.0, 600.0, 30.0, 1)
        elif algo == "NN":
            self._add_spin("start", 0, 150, 0)
            self._add_check("multistart", False)
        elif algo == "HC":
            self._add_spin("max_iterations", 100, 100000, 5000)
            self._add_spin("restarts", 1, 200, 10)
        elif algo == "SA":
            self._add_float("initial_temp", 1.0, 50000.0, 1000.0, 0)
            self._add_float("cooling_rate", 0.8, 0.9999, 0.995, 4)
            self._add_float("min_temp", 0.0001, 10.0, 0.01, 4)
            self._add_spin("max_iterations", 100, 100000, 5000)
        elif algo == "GA":
            self._add_spin("population_size", 10, 500, 80)
            self._add_spin("generations", 10, 2000, 150)
            self._add_float("mutation_rate", 0.01, 1.0, 0.15, 2)
            self._add_float("crossover_rate", 0.1, 1.0, 0.85, 2)
            self._add_spin("elitism", 0, 20, 2)

    def _add_spin(self, name: str, lo: int, hi: int, val: int):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        self.params_layout.addRow(name, w)
        self._param_widgets[name] = w

    def _add_float(self, name: str, lo: float, hi: float, val: float, decimals: int):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(val)
        self.params_layout.addRow(name, w)
        self._param_widgets[name] = w

    def _add_combo(self, name: str, items: list[str], index: int):
        w = QComboBox()
        w.addItems(items)
        w.setCurrentIndex(index)
        self.params_layout.addRow(name, w)
        self._param_widgets[name] = w

    def _add_check(self, name: str, checked: bool):
        w = QCheckBox()
        w.setChecked(checked)
        self.params_layout.addRow(name, w)
        self._param_widgets[name] = w

    def _current_params(self, algo: str):
        w = self._param_widgets
        if algo == "BKT":
            return BKTParams(
                mode=w["mode"].currentText(),
                y_solutions=w["y_solutions"].value(),
                time_limit_s=w["time_limit_s"].value(),
            )
        if algo == "NN":
            return NNParams(start=w["start"].value(), multistart=w["multistart"].isChecked())
        if algo == "HC":
            return HCParams(
                max_iterations=w["max_iterations"].value(),
                restarts=w["restarts"].value(),
                seed=self.seed_spin.value(),
            )
        if algo == "SA":
            return SAParams(
                initial_temp=w["initial_temp"].value(),
                cooling_rate=w["cooling_rate"].value(),
                min_temp=w["min_temp"].value(),
                max_iterations=w["max_iterations"].value(),
                seed=self.seed_spin.value(),
            )
        if algo == "GA":
            return GAParams(
                population_size=w["population_size"].value(),
                generations=w["generations"].value(),
                mutation_rate=w["mutation_rate"].value(),
                crossover_rate=w["crossover_rate"].value(),
                elitism=w["elitism"].value(),
                seed=self.seed_spin.value(),
            )
        return None

    def _generate_cities(self):
        n = self.spin_n.value()
        self.cities = generate_cities(n, seed=self.seed_spin.value())
        self.dist_matrix = matrix_from_coords(self.cities)
        self._draw_path([], "Orașe generate (Euclidian)")
        msg = f"Generate {n} orașe în coordonate 2D (Euclidian)."
        self.result_text.setPlainText(msg); self.console_label.setText(msg)

    def _generate_non_euclidian(self):
        n = self.spin_n.value()
        self.dist_matrix = random_matrix(n, seed=self.seed_spin.value())
        self._layout_cities_circle(n)
        self._draw_path([], f"Generat matrice non-Euclidiană N={n}")
        msg = f"Generat matrice non-Euclidiană de dimensiune {n}x{n}.\nNotă: Orașele sunt dispuse circular pentru vizualizare."
        self.result_text.setPlainText(msg); self.console_label.setText(msg)

    def _layout_cities_circle(self, n: int):
        self.cities = []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            self.cities.append((400.0 + 220.0 * math.cos(angle), 300.0 + 220.0 * math.sin(angle)))

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Deschide matrice TSP", "", "Text (*.txt);;All (*)")
        if not path:
            return
        n, matrix = load_matrix_file(path)
        self.dist_matrix = matrix
        self._layout_cities_circle(n)
        self.spin_n.setValue(n)
        self._draw_path([], f"Încărcat matrice N={n}")
        fname = os.path.basename(path)
        msg = f"Încărcat fișierul: {fname}\nDimensiune: N = {n} orașe"
        self.result_text.setPlainText(msg); self.console_label.setText(msg)

    def _save_file(self):
        if not self.dist_matrix:
            msg = "Nu exista nicio matrice de salvat!"
            self.result_text.setPlainText(msg); self.console_label.setText(msg)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvează matrice TSP", "", "Text (*.txt);;All (*)")
        if not path:
            return
        save_matrix_file(path, self.dist_matrix)
        fname = os.path.basename(path)
        msg = f"Salvat cu succes: {fname}"
        self.result_text.setPlainText(msg); self.console_label.setText(msg)

    def _draw_path(self, path: list[int], title: str):
        self.canvas_map.clear()
        ax = self.canvas_map.ax
        if self.cities:
            xs = [c[0] for c in self.cities]
            ys = [c[1] for c in self.cities]
            ax.scatter(xs, ys, c="crimson", s=45, zorder=3)
            for i, (x, y) in enumerate(self.cities):
                ax.annotate(str(i), (x, y), fontsize=9, fontweight="bold", xytext=(4, 4), textcoords="offset points")
        if path:
            px = [self.cities[i][0] for i in path] + [self.cities[path[0]][0]]
            py = [self.cities[i][1] for i in path] + [self.cities[path[0]][1]]
            ax.plot(px, py, "b-", linewidth=2.0, zorder=2)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        self.canvas_map.refresh()
        self.viz_tabs.setCurrentIndex(0)  # Switch to Map Tab

    def _set_busy(self, busy: bool):
        self.btn_run.setEnabled(not busy)
        self.btn_compare.setEnabled(not busy)
        self.btn_bench.setEnabled(not busy)
        self.btn_gen.setEnabled(not busy)
        self.btn_gen_non_eucl.setEnabled(not busy)
        self.btn_load.setEnabled(not busy)
        self.btn_save.setEnabled(not busy)
        if not busy:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Gata.")

    def _on_progress(self, current: int, total: int):
        if total == 0:
            # Indeterminate — animated bar, no percentage
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Se ruleaza...")
            self.progress_label.setText("Se ruleaza...")
        else:
            pct = int(current / total * 100)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}%")
            self.progress_label.setText(f"Pas {current} / {total}  ({pct}%)")

    def _run_async(self, fn, on_ok):
        if self._worker and self._worker.isRunning():
            return
        self._set_busy(True)
        self.progress_bar.setVisible(True)
        self._worker = WorkerThread(fn)
        self._worker.finished_ok.connect(on_ok)
        self._worker.failed.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(lambda: self._set_busy(False))
        self._worker.start()

    def _on_error(self, msg: str):
        err = f"Eroare: {msg}"
        self.result_text.setPlainText(err); self.console_label.setText(err)

    def _run_single(self):
        if not self.dist_matrix:
            return
        algo = self.algo_combo.currentText()
        params = self._current_params(algo)
        matrix = self.dist_matrix

        def job():
            # Indeterminate — we don't know exact duration
            return algo, SOLVERS[algo](matrix, params)

        def done(payload):
            algo_name, result = payload
            self._show_result(algo_name, result)

        self._on_progress(0, 0)  # Start indeterminate
        self._run_async(job, done)

    def _show_result(self, algo: str, result: TSPResult):
        # Format route as: 0 -> 3 -> 7 -> ... -> 0
        if result.path:
            route_str = " -> ".join(str(c) for c in result.path) + f" -> {result.path[0]}"
        else:
            route_str = "N/A"
        
        meta_parts = [f"{k}: {v}" for k, v in result.meta.items() if k != "history"]
        meta_str = "  |  ".join(meta_parts)
        
        # Plain text for console
        console_text = (
            f"Algoritm: {algo}\n"
            f"Cost:     {result.cost:.4f}\n"
            f"Timp:     {result.elapsed_s:.5f} s\n"
            f"Rută:     {route_str}\n"
            f"Meta:     {meta_str}"
        )
        self.result_text.setPlainText(console_text); self.console_label.setText(console_text)
        self._draw_path(result.path, f"{algo} — cost {result.cost:.2f}")

        # Draw convergence plot if history is available (SA, GA, HC)
        history = result.meta.get("history")
        if history:
            self.canvas_conv.clear()
            ax = self.canvas_conv.ax
            ax.plot(range(len(history)), history, color="crimson", linewidth=2.0, label="Cost curent")
            ax.set_title(f"Convergență - {algo}")
            ax.set_xlabel("Eșantion Iterații / Generații")
            ax.set_ylabel("Cel mai bun cost")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            self.canvas_conv.refresh()
            # Switch to the convergence tab automatically
            self.viz_tabs.setCurrentIndex(1)
        else:
            self.canvas_conv.clear()
            self.canvas_conv.ax.set_title("Nicio convergență (doar pentru SA, GA, HC)")
            self.canvas_conv.refresh()

    def _run_compare(self):
        if not self.dist_matrix:
            return
        matrix = self.dist_matrix
        n = len(matrix)

        algos_to_run = [
            (algo, fn) for algo, fn in SOLVERS.items()
            if not (algo == "BKT" and n > 20)
        ]
        total = len(algos_to_run)

        # mutable counter accessible from worker thread via closure
        counter = [0]

        def job():
            rows = []
            for i, (algo, fn) in enumerate(algos_to_run):
                params = None
                if algo == "NN":
                    params = NNParams(multistart=True)
                elif algo == "BKT":
                    params = BKTParams(mode="timp", time_limit_s=5.0)
                r = fn(matrix, params) if params else fn(matrix)
                rows.append((algo, r))
                counter[0] = i + 1
                self._worker.progress.emit(i + 1, total)
            return rows

        def done(rows):
            self.canvas_comp.clear()
            fig = self.canvas_comp.fig
            fig.clear()
            
            names = [a for a, _ in rows]
            costs = [r.cost for _, r in rows]
            times = [r.elapsed_s for _, r in rows]

            # 1. Bar plot on the left (Costs)
            ax1 = fig.add_subplot(121)
            x = range(len(names))
            ax1.bar(x, costs, color="steelblue", alpha=0.8, edgecolor="black", width=0.5)
            ax1.set_xticks(list(x))
            ax1.set_xticklabels(names, rotation=25)
            ax1.set_ylabel("Costul Turului (Valoare)")
            ax1.set_title("Calitate Soluție (Cost)")
            ax1.grid(True, linestyle="--", alpha=0.3)

            # Add labels above bars
            for i, c in enumerate(costs):
                ax1.text(i, c + (max(costs) * 0.01), f"{c:.1f}", ha="center", fontsize=8, fontweight="bold")

            # 2. Performance (Cost vs Time scatter plot) on the right
            ax2 = fig.add_subplot(122)
            for name, c, t in zip(names, costs, times):
                ax2.scatter(t, c, s=180, label=name, marker="o", edgecolors="black", alpha=0.95)
            ax2.set_xlabel("Timp Execuție (s)")
            ax2.set_ylabel("Cost")
            ax2.set_title("Eficiență: Cost vs Timp")
            ax2.legend(loc="best")
            ax2.grid(True, linestyle="--", alpha=0.5)

            self.canvas_comp.refresh()
            self.viz_tabs.setCurrentIndex(2)  # Switch to Comparison Tab

            best_algo = min(rows, key=lambda t: t[1].cost)[0]

            # Plain console text
            console_lines = [f"Comparare Algoritmi (N={n}):", "-" * 40]
            for a, r in rows:
                best_mark = " [CEL MAI BUN]" if a == best_algo else ""
                console_lines.append(f"{a:<6}  cost={r.cost:.2f}  timp={r.elapsed_s:.5f}s{best_mark}")
            self.result_text.setPlainText("\n".join(console_lines))
            self.console_label.setText("\n".join(console_lines))
            best = min(rows, key=lambda t: t[1].cost)
            self._draw_path(best[1].path, f"Cel mai bun: {best[0]}")

        self._run_async(job, done)

    def _run_benchmark(self):
        try:
            sizes = [int(s.strip()) for s in self.line_sizes.text().split(",") if s.strip()]
        except Exception:
            msg = "Format dimensiuni invalid! Foloseste valori separate prin virgula."
            self.result_text.setPlainText(msg); self.console_label.setText(msg)
            return

        repeats = self.spin_repeats.value()
        seed = self.seed_spin.value()

        # ~5 algos per N (BKT only for small N, but approximate is fine for display)
        total_steps = 5 * len(sizes)

        def job():
            from src.tsp.benchmarks import run_tsp_benchmark as _bench
            # Run using the official benchmark function (correct random_matrix + params)
            # We wrap it to emit progress after each completed (n, algo) pair
            step = [0]
            all_rows = []
            for n in sizes:
                partial = _bench(sizes=[n], repeats=repeats, seed=seed)
                all_rows.extend(partial)
                step[0] += len(partial)  # one row per algo per n
                self._worker.progress.emit(step[0], total_steps)
            return all_rows

        def done(rows):
            self.canvas_bench.clear()
            fig = self.canvas_bench.fig
            fig.clear()

            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            algos = sorted({r.algorithm for r in rows})
            for algo in algos:
                subset = [r for r in rows if r.algorithm == algo]
                subset.sort(key=lambda r: r.n)
                ns = [r.n for r in subset]

                cost_means = [r.cost_mean for r in subset]
                cost_stds = [r.cost_std for r in subset]
                time_means = [r.time_mean for r in subset]
                time_stds = [r.time_std for r in subset]

                # Execution Time vs N (Log scale)
                ax1.errorbar(ns, time_means, yerr=time_stds, fmt="-o", capsize=4, label=algo, linewidth=1.8)
                # Cost vs N
                ax2.errorbar(ns, cost_means, yerr=cost_stds, fmt="-s", capsize=4, label=algo, linewidth=1.8)

            ax1.set_xlabel("Dimensiune Problemă N (Orașe)")
            ax1.set_ylabel("Timp mediu de rulare (s)")
            ax1.set_yscale("log")
            ax1.set_title("Timp Mediu vs N (Scală Log)")
            ax1.legend(loc="upper left")
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.set_xlabel("Dimensiune Problemă N (Orașe)")
            ax2.set_ylabel("Cost Mediu Soluție")
            ax2.set_title("Performanță Cost Mediu vs N")
            ax2.legend(loc="upper left")
            ax2.grid(True, linestyle="--", alpha=0.5)

            self.canvas_bench.refresh()
            self.viz_tabs.setCurrentIndex(3)  # Switch to Benchmark Tab

            algos_sorted = sorted({r.algorithm for r in rows})
            sizes_sorted = sorted({r.n for r in rows})

            # Plain console text
            console_lines = [
                f"Benchmark finalizat (repetări={repeats}):",
                f"{'Algoritm':<8}" + "".join(f" {'N='+str(n):<12}" for n in sizes_sorted),
                "-" * (8 + 12 * len(sizes_sorted))
            ]
            for algo in algos_sorted:
                algo_rows = {r.n: r for r in rows if r.algorithm == algo}
                line = f"{algo:<8}"
                for n in sizes_sorted:
                    if n in algo_rows:
                        r = algo_rows[n]
                        line += f" {r.cost_mean:.1f}/{r.time_mean:.3f}s  "
                    else:
                        line += " -           "
                console_lines.append(line)
            self.result_text.setPlainText("\n".join(console_lines)); self.console_label.setText("\n".join(console_lines))

            # console_label already updated with plain text above

        self._run_async(job, done)
