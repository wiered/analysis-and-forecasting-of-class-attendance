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

from src.model import (
    AttendancePredictor,
    RescheduleEffect,
    BestSlotResult,
    WEEKDAYS,
    TIME_SLOTS,
)

from .charts import plot_reschedule_attendance, plot_reschedule_risk
from .workers import RescheduleEffectWorker, BestSlotWorker, run_worker

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

        btn_calc = QPushButton("Рассчитать эффект переноса")
        btn_calc.clicked.connect(self._on_calc)
        layout.addWidget(btn_calc)

        btn_best = QPushButton("Подобрать лучший слот")
        btn_best.clicked.connect(self._on_best_slot)
        layout.addWidget(btn_best)

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
        self._last_lesson_id = lesson_id
        self._last_new_weekday = new_weekday
        self._last_new_time_slot = new_time_slot

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

    def _on_best_slot(self) -> None:
        if not self.predictor:
            self.status_label.setText("Предиктор недоступен.")
            return
        group = self.group_spin.value()
        lesson_id = self.lesson_spin.value()
        lesson_date = self.date_edit.date().toString("yyyy-MM-dd")
        self.status_label.setText("Поиск лучшего слота…")
        self.result_frame.setVisible(False)
        self._worker = BestSlotWorker(self.predictor, group, lesson_id, lesson_date)
        self._thread = run_worker(
            self._worker,
            on_finished=self._on_best_slot_result,
            on_error=self._on_error,
        )
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: setattr(self, "_worker", None))

    def _on_best_slot_result(self, result: BestSlotResult) -> None:
        self.weekday_combo.setCurrentText(result.best_weekday)
        self.time_combo.setCurrentText(result.best_time_slot)
        self.status_label.setText("")
        from_slot = None
        if result.current_weekday and result.current_time_slot:
            from_slot = f"{result.current_weekday} {result.current_time_slot}"
        to_slot = f"{result.best_weekday} {result.best_time_slot}"
        self._on_result(result.reschedule_effect, from_slot=from_slot, to_slot=to_slot)

    def _on_result(
        self,
        effect: RescheduleEffect,
        from_slot: Optional[str] = None,
        to_slot: Optional[str] = None,
    ) -> None:
        self.status_label.setText("")
        if from_slot is None and to_slot is None and getattr(self, "_last_lesson_id", None) and self.predictor:
            try:
                current = self.predictor.get_lesson_schedule(self._last_lesson_id)
                if current:
                    from_slot = f"{current[0]} {current[1]}"
                to_slot = f"{getattr(self, '_last_new_weekday', '')} {getattr(self, '_last_new_time_slot', '')}".strip()
            except Exception:
                pass
        lines = []
        if from_slot and to_slot:
            lines.append(f"Перенос: {from_slot} → {to_slot}")
            lines.append("")
        delta_sign = "+" if effect.delta >= 0 else ""
        lines.extend([
            "До переноса:",
            f"  Средняя вероятность посещения: {effect.avg_attendance_before:.2%}",
            f"  Доля в зоне риска: {effect.risk_pct_before}%",
            "",
            "После переноса:",
            f"  Средняя вероятность посещения: {effect.avg_attendance_after:.2%}",
            f"  Доля в зоне риска: {effect.risk_pct_after}%",
            "",
            f"Изменение средней посещаемости: {delta_sign}{effect.delta:.2%}",
        ])
        self.result_label.setText("\n".join(lines))
        plot_reschedule_attendance(self.plot_att.getPlotItem(), effect)
        plot_reschedule_risk(self.plot_risk.getPlotItem(), effect)
        self.result_frame.setVisible(True)

    def _on_error(self, msg: str) -> None:
        logger.warning("Сценарный анализ: ошибка — %s", msg)
        self.status_label.setText(f"Ошибка: {msg}")
        self.result_frame.setVisible(False)
