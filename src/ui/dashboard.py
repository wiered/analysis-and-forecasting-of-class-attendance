"""Виджет дашборда текущих показателей по группе."""
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
    QFrame,
)
from PySide6.QtCore import QDate
from PySide6.QtGui import QFont

from src.model import AttendancePredictor, GroupSummary, StudentPrediction
from src.utils import FEATURE_MAP

from .charts import plot_probability_histogram, plot_top_factors
from .workers import PredictGroupWorker, run_worker

logger = logging.getLogger(__name__)


class DashboardWidget(QWidget):
    def __init__(self, predictor: Optional[AttendancePredictor] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.predictor = predictor
        self._thread = None

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

        btn = QPushButton("Загрузить показатели")
        btn.clicked.connect(self._on_load)
        layout.addWidget(btn)

        self.status_label = QLabel("Введите параметры и нажмите «Загрузить показатели».")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.results_frame = QFrame()
        self.results_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        results_layout = QVBoxLayout(self.results_frame)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setFont(QFont("", 10))
        self.factors_label = QLabel("")
        self.factors_label.setWordWrap(True)
        results_layout.addWidget(self.summary_label)
        results_layout.addWidget(self.factors_label)
        self.plot_hist = pg.PlotWidget(title="")
        self.plot_factors = pg.PlotWidget(title="")
        results_layout.addWidget(self.plot_hist)
        results_layout.addWidget(self.plot_factors)
        self.results_frame.setVisible(False)
        layout.addWidget(self.results_frame)

        layout.addStretch()

    def _on_load(self) -> None:
        logger.debug("Дашборд: нажата кнопка «Загрузить показатели»")
        if not self.predictor:
            logger.warning("Дашборд: предиктор отсутствует, загрузка отменена")
            self.status_label.setText("Предиктор недоступен. Проверьте БД и файлы модели.")
            return
        group = self.group_spin.value()
        lesson_id = self.lesson_spin.value()
        qdate = self.date_edit.date()
        lesson_date = qdate.toString("yyyy-MM-dd")
        logger.info("Дашборд: запрос показателей group=%s lesson_id=%s lesson_date=%s", group, lesson_id, lesson_date)

        self.status_label.setText("Загрузка…")
        self.results_frame.setVisible(False)

        self._worker = PredictGroupWorker(self.predictor, group, lesson_id, lesson_date)
        logger.debug("Дашборд: воркер создан, запуск потока")
        self._thread = run_worker(
            self._worker,
            on_finished=self._on_results,
            on_error=self._on_error,
        )
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: setattr(self, "_worker", None))
        logger.debug("Дашборд: поток запущен, ожидание результата")

    def _on_results(self, results: List[StudentPrediction]) -> None:
        try:
            if not results:
                self.status_label.setText("В группе нет студентов.")
                self.results_frame.setVisible(False)
                return
            summary = AttendancePredictor.group_summary_from_predictions(
                results, self.group_spin.value(), self.lesson_spin.value(),
                self.date_edit.date().toString("yyyy-MM-dd"),
            )
            logger.info("Дашборд: получена сводка, студентов=%s, в зоне риска=%s", summary.total_students, summary.students_at_risk)
            self.status_label.setText("")
            self.summary_label.setText(
                f"Группа {summary.group}, занятие {summary.lesson_id}, дата {summary.lesson_date}\n"
                f"Студентов: {summary.total_students}\n"
                f"Средняя вероятность посещения: {summary.avg_attendance_probability:.2%}\n"
                f"Студентов в зоне риска (вероятность пропуска > 50%): {summary.students_at_risk} ({summary.risk_percentage}%)"
            )
            factors_text = "Топ факторов по группе:\n"
            for gf in summary.top_group_factors:
                readable = FEATURE_MAP.get(gf.feature, gf.feature)
                factors_text += (
                    f"  • {readable}: затронуто студентов {gf.students_affected}, "
                    f"средний вклад {gf.avg_impact:.3f}\n"
                )
            self.factors_label.setText(factors_text)
            probs = [r.attendance_probability for r in results]
            plot_probability_histogram(self.plot_hist.getPlotItem(), probs)
            plot_top_factors(self.plot_factors.getPlotItem(), summary.top_group_factors)
            self.results_frame.setVisible(True)
        except Exception as e:
            logger.exception("Дашборд: ошибка при отображении сводки: %s", e)

    def _on_error(self, msg: str) -> None:
        logger.warning("Дашборд: получен сигнал ошибки — %s", msg)
        self.status_label.setText(f"Ошибка: {msg}")
        self.results_frame.setVisible(False)
