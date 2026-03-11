"""Виджет индикаторов риска — таблица студентов в зоне риска."""
import logging
from typing import Optional, List

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QDateEdit,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import QDate

from src.model import AttendancePredictor, StudentPrediction

from .charts import plot_risk_split
from .workers import PredictGroupWorker, run_worker

logger = logging.getLogger(__name__)


class RiskIndicatorsWidget(QWidget):
    def __init__(self, predictor: Optional[AttendancePredictor] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.predictor = predictor
        self._thread = None
        self._worker = None

        layout = QVBoxLayout(self)

        form = QFormLayout()
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
        form.addRow("Дата занятия:", self.date_edit)

        layout.addLayout(form)

        btn = QPushButton("Показать студентов в зоне риска")
        btn.clicked.connect(self._on_load)
        layout.addWidget(btn)

        self.status_label = QLabel("Введите параметры и нажмите кнопку.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.plot_risk = pg.PlotWidget(title="")
        layout.addWidget(self.plot_risk)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ФИО", "ID", "Вероятность пропуска", "Вероятность посещения"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

    def _on_load(self) -> None:
        if not self.predictor:
            self.status_label.setText("Предиктор недоступен. Проверьте БД и файлы модели.")
            return
        group = self.group_spin.value()
        lesson_id = self.lesson_spin.value()
        lesson_date = self.date_edit.date().toString("yyyy-MM-dd")

        self.status_label.setText("Загрузка…")
        self.table.setRowCount(0)

        self._worker = PredictGroupWorker(self.predictor, group, lesson_id, lesson_date)
        self._thread = run_worker(
            self._worker,
            on_finished=self._on_results,
            on_error=self._on_error,
        )
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: setattr(self, "_worker", None))

    def _on_results(self, results: List[StudentPrediction]) -> None:
        self.status_label.setText("")
        at_risk_list = [r for r in results if r.absence_probability > 0.5]
        at_risk_count = len(at_risk_list)
        not_at_risk_count = len(results) - at_risk_count
        plot_risk_split(self.plot_risk.getPlotItem(), at_risk_count, not_at_risk_count)
        self.table.setRowCount(len(at_risk_list))
        for i, pred in enumerate(at_risk_list):
            name = pred.full_name or "—"
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(str(pred.student_id)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{pred.absence_probability:.2%}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{pred.attendance_probability:.2%}"))
        if not at_risk_list:
            self.status_label.setText("Нет студентов в зоне риска (вероятность пропуска > 50%).")

    def _on_error(self, msg: str) -> None:
        logger.warning("Индикаторы риска: ошибка — %s", msg)
        self.status_label.setText(f"Ошибка: {msg}")
        self.table.setRowCount(0)
