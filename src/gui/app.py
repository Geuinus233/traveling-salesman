"""PySide6 application entry point with team info dialogue and premium styling."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.nlp_tab import NLPTab
from src.gui.tsp_tab import TSPTab
from src.system_info import get_machine_info


class TeamInfoDialog(QDialog):
    """Custom premium dialog to present team members, roles, and avatar images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Informații Echipă")
        self.setFixedSize(520, 360)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a24;
                color: #f3f4f6;
            }
            QLabel {
                color: #f3f4f6;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)

        title = QLabel("👥 Echipa Antigravity")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Disciplina: Inteligență Artificială — Anul 3, AC")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 11, QFont.StyleItalic))
        subtitle.setStyleSheet("color: #9ca3af; margin-bottom: 15px;")
        layout.addWidget(subtitle)

        members_layout = QHBoxLayout()
        members_layout.setSpacing(20)
        layout.addLayout(members_layout)

        # Member 1: Andrei
        m1_layout = QVBoxLayout()
        m1_img = QLabel()
        m1_img.setAlignment(Qt.AlignCenter)
        img_path_andrei = os.path.join("data", "team", "andrei.png")
        if os.path.exists(img_path_andrei):
            pix = QPixmap(img_path_andrei).scaled(
                110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            m1_img.setPixmap(pix)
        else:
            m1_img.setText("[Andrei Portrait]")
            m1_img.setStyleSheet(
                "background-color: #374151; border-radius: 55px; min-width: 110px; min-height: 110px; color: #9ca3af;"
            )
        m1_layout.addWidget(m1_img)

        m1_name = QLabel("<b>Andrei Popescu</b>")
        m1_name.setAlignment(Qt.AlignCenter)
        m1_name.setFont(QFont("Segoe UI", 11))
        m1_role = QLabel("TSP Solvers & Stats")
        m1_role.setAlignment(Qt.AlignCenter)
        m1_role.setFont(QFont("Segoe UI", 9))
        m1_role.setStyleSheet("color: #9ca3af;")
        m1_layout.addWidget(m1_name)
        m1_layout.addWidget(m1_role)
        members_layout.addLayout(m1_layout)

        # Separator line
        sep = QLabel()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #374151;")
        members_layout.addWidget(sep)

        # Member 2: Elena
        m2_layout = QVBoxLayout()
        m2_img = QLabel()
        m2_img.setAlignment(Qt.AlignCenter)
        img_path_elena = os.path.join("data", "team", "elena.png")
        if os.path.exists(img_path_elena):
            pix = QPixmap(img_path_elena).scaled(
                110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            m2_img.setPixmap(pix)
        else:
            m2_img.setText("[Elena Portrait]")
            m2_img.setStyleSheet(
                "background-color: #374151; border-radius: 55px; min-width: 110px; min-height: 110px; color: #9ca3af;"
            )
        m2_layout.addWidget(m2_img)

        m2_name = QLabel("<b>Elena Dumitrescu</b>")
        m2_name.setAlignment(Qt.AlignCenter)
        m2_name.setFont(QFont("Segoe UI", 11))
        m2_role = QLabel("NLP Classifiers & Lexicons")
        m2_role.setAlignment(Qt.AlignCenter)
        m2_role.setFont(QFont("Segoe UI", 9))
        m2_role.setStyleSheet("color: #9ca3af;")
        m2_layout.addWidget(m2_name)
        m2_layout.addWidget(m2_role)
        members_layout.addLayout(m2_layout)

        layout.addSpacing(15)

        # Close button
        btn_close = QPushButton("Închide")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, 0, Qt.AlignCenter)


class MainWindow(QMainWindow):
    """Main window with an elegant header bar containing team details trigger."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IA — TSP & NLP Laboratory")
        self.resize(1200, 820)

        # Apply a clean, modern stylesheet to the main application elements
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f8fafc;
            }
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                background-color: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #e2e8f0;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #1e3a8a;
                border: 1px solid #e2e8f0;
                border-bottom: 1px solid white;
                font-weight: bold;
            }
        """
        )

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 12, 15, 12)
        main_layout.setSpacing(10)

        # Header Bar
        header = QHBoxLayout()
        title_label = QLabel("<b>TSP & NLP Laboratory Dashboard</b>")
        title_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title_label.setStyleSheet("color: #1e3a8a; font-family: 'Segoe UI', sans-serif;")
        header.addWidget(title_label)
        header.addStretch()

        btn_team = QPushButton("👥 Informații Echipă")
        btn_team.setFont(QFont("Segoe UI", 10, QFont.Bold))
        btn_team.setStyleSheet(
            """
            QPushButton {
                background-color: #1e3a8a;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """
        )
        btn_team.clicked.connect(self.show_team_dialog)
        header.addWidget(btn_team)
        main_layout.addLayout(header)

        # Tabs Layout
        tabs = QTabWidget()
        tabs.addTab(TSPTab(), "TSP (BKT, NN, HC, SA, GA)")
        tabs.addTab(NLPTab(), "NLP Classification")

        about = QWidget()
        about_layout = QVBoxLayout(about)
        info = get_machine_info()
        about_layout.addWidget(
            QLabel(
                "<h3>Runtime environment</h3>"
                f"<pre>{info.as_text()}</pre>"
                "<p>All experiment outputs include machine metadata for reproducibility.</p>"
                "<p><b>TSP:</b> Backtracking (BKT), Nearest Neighbor, Hill Climbing, "
                "Simulated Annealing, Genetic Algorithm.</p>"
                "<p><b>NLP:</b> TF-IDF pipeline with Naive Bayes, SVM, Logistic Regression, "
                "Random Forest on English datasets.</p>"
            )
        )
        about_layout.addStretch()
        tabs.addTab(about, "Machine info")

        main_layout.addWidget(tabs)

    def show_team_dialog(self):
        dialog = TeamInfoDialog(self)
        dialog.exec()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
