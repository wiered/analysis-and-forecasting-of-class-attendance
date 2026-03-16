"""Вкладка «Симуляция»: симуляция поведения агентов за 1 неделю с логами."""
import logging
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QDateEdit,
    QLabel,
    QPlainTextEdit,
    QGroupBox,
)
from PySide6.QtCore import QDate

from src.model import AttendancePredictor

from .workers import SimulationWorker, run_worker

logger = logging.getLogger(__name__)


class SimulationWidget(QWidget):
    """
    Виджет симуляции: при нажатии «Старт» выполняется симуляция 1 недели (5 рабочих дней).
    В логе отображаются решения студентов (идти/пропустить), политика преподавателя
    и решения деканата (расписание, аудитория, уведомления).
    """

    def __init__(self, predictor: Optional[AttendancePredictor] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.predictor = predictor
        self._thread = None
        self._worker = None

        layout = QVBoxLayout(self)

        params = QGroupBox("Параметры симуляции")
        form = QFormLayout(params)
        self.group_spin = QSpinBox()
        self.group_spin.setRange(1, 999)
        self.group_spin.setValue(1)
        form.addRow("Группа:", self.group_spin)

        self.lesson_spin = QSpinBox()
        self.lesson_spin.setRange(1, 9999)
        self.lesson_spin.setValue(2)
        form.addRow("ID занятия:", self.lesson_spin)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        form.addRow("Начало недели:", self.date_edit)

        layout.addWidget(params)

        self.start_btn = QPushButton("Старт (1 неделя)")
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)

        self.status_label = QLabel("Задайте параметры и нажмите «Старт» для запуска симуляции.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("Лог симуляции:"))
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("Здесь появится лог решений агентов по дням.")
        layout.addWidget(self.log_edit)

        if not predictor:
            self.start_btn.setEnabled(False)
            self.status_label.setText("Предиктор недоступен. Симуляция невозможна.")

    def _on_start(self) -> None:
        if not self.predictor:
            return
        group = self.group_spin.value()
        lesson_id = self.lesson_spin.value()
        start_date = self.date_edit.date().toPython()

        self.start_btn.setEnabled(False)
        self.status_label.setText("Идёт симуляция недели…")
        self.log_edit.clear()

        self._worker = SimulationWorker(
            self.predictor,
            group=group,
            lesson_id=lesson_id,
            start_date=start_date,
        )
        self._thread = run_worker(
            self._worker,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: setattr(self, "_worker", None))

    def _on_finished(self, log_text: str) -> None:
        self.log_edit.setPlainText(log_text)
        self.status_label.setText("Симуляция завершена.")
        self.start_btn.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self.status_label.setText(f"Ошибка: {msg}")
        self.log_edit.setPlainText(self.log_edit.toPlainText() + f"\n[Ошибка] {msg}")
        self.start_btn.setEnabled(True)
