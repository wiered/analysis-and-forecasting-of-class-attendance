"""Виджет сценарного анализа — эффект переноса занятия."""
import logging
from typing import Optional

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QDateEdit,
    QLabel,
    QComboBox,
    QFrame,
)
from PySide6.QtCore import QDate

from src.model import AttendancePredictor, RescheduleEffect, WEEKDAYS, TIME_SLOTS

from .charts import plot_reschedule_attendance, plot_reschedule_risk
from .workers import RescheduleEffectWorker, run_worker

logger = logging.getLogger(__name__)


class ScenarioWidget(QWidget):
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

        self.weekday_combo = QComboBox()
        self.weekday_combo.addItems(WEEKDAYS)
        self.weekday_combo.setCurrentText("Wednesday")
        form.addRow("Новый день недели:", self.weekday_combo)

        self.time_combo = QComboBox()
        self.time_combo.addItems(TIME_SLOTS)
        self.time_combo.setCurrentText("10:30")
        form.addRow("Новое время:", self.time_combo)

        layout.addLayout(form)

        btn = QPushButton("Рассчитать эффект переноса")
        btn.clicked.connect(self._on_calc)
        layout.addWidget(btn)

        self.status_label = QLabel("Введите параметры и нажмите кнопку.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.result_frame = QFrame()
        self.result_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        result_layout = QVBoxLayout(self.result_frame)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        self.plot_att = pg.PlotWidget(title="")
        self.plot_risk = pg.PlotWidget(title="")
        result_layout.addWidget(self.plot_att)
        result_layout.addWidget(self.plot_risk)
        self.result_frame.setVisible(False)
        layout.addWidget(self.result_frame)

        layout.addStretch()

    def _on_calc(self) -> None:
        if not self.predictor:
            self.status_label.setText("Предиктор недоступен.")
            return
        group = self.group_spin.value()
        lesson_id = self.lesson_spin.value()
        lesson_date = self.date_edit.date().toString("yyyy-MM-dd")
        new_weekday = self.weekday_combo.currentText()
        new_time_slot = self.time_combo.currentText()

        self.status_label.setText("Расчёт…")
        self.result_frame.setVisible(False)

        self._worker = RescheduleEffectWorker(
            self.predictor, group, lesson_id, lesson_date, new_weekday, new_time_slot
        )
        self._thread = run_worker(
            self._worker,
            on_finished=self._on_result,
            on_error=self._on_error,
        )
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: setattr(self, "_worker", None))

    def _on_result(self, effect: RescheduleEffect) -> None:
        self.status_label.setText("")
        delta_sign = "+" if effect.delta >= 0 else ""
        self.result_label.setText(
            "До переноса:\n"
            f"  Средняя вероятность посещения: {effect.avg_attendance_before:.2%}\n"
            f"  Доля в зоне риска: {effect.risk_pct_before}%\n\n"
            "После переноса:\n"
            f"  Средняя вероятность посещения: {effect.avg_attendance_after:.2%}\n"
            f"  Доля в зоне риска: {effect.risk_pct_after}%\n\n"
            f"Изменение средней посещаемости: {delta_sign}{effect.delta:.2%}"
        )
        plot_reschedule_attendance(self.plot_att.getPlotItem(), effect)
        plot_reschedule_risk(self.plot_risk.getPlotItem(), effect)
        self.result_frame.setVisible(True)

    def _on_error(self, msg: str) -> None:
        logger.warning("Сценарный анализ: ошибка — %s", msg)
        self.status_label.setText(f"Ошибка: {msg}")
        self.result_frame.setVisible(False)
