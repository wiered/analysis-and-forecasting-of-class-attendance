"""Виджет модуля прогнозирования — по студенту или по группе."""
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
    QGroupBox,
    QRadioButton,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtCore import QDate

from src.model import AttendancePredictor, StudentPrediction
from src.utils import FEATURE_MAP

from .charts import (
    plot_probability_histogram,
    plot_student_ranking,
    plot_student_factors,
)
from .workers import (
    PredictStudentWorker,
    PredictGroupWorker,
    run_worker,
)

logger = logging.getLogger(__name__)


class ForecastWidget(QWidget):
    def __init__(self, predictor: Optional[AttendancePredictor] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.predictor = predictor
        self._thread = None
        self._worker = None

        layout = QVBoxLayout(self)

        mode_group = QGroupBox("Режим")
        mode_layout = QVBoxLayout(mode_group)
        self.radio_student = QRadioButton("Прогноз по одному студенту")
        self.radio_group = QRadioButton("Прогноз по группе")
        self.radio_student.setChecked(True)
        mode_layout.addWidget(self.radio_student)
        mode_layout.addWidget(self.radio_group)
        layout.addWidget(mode_group)

        form = QFormLayout()
        self.student_spin = QSpinBox()
        self.student_spin.setRange(1, 99999)
        self.student_spin.setValue(1)
        form.addRow("ID студента:", self.student_spin)

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

        btn = QPushButton("Рассчитать прогноз")
        btn.clicked.connect(self._on_calc)
        layout.addWidget(btn)

        self.status_label = QLabel("Выберите режим и параметры.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.student_result = QLabel("")
        self.student_result.setWordWrap(True)
        self.student_result.setFrameStyle(QFrame.Shape.StyledPanel)
        layout.addWidget(self.student_result)
        self.plot_student_factors = pg.PlotWidget(title="")
        layout.addWidget(self.plot_student_factors)

        self.group_table = QTableWidget()
        self.group_table.setColumnCount(5)
        self.group_table.setHorizontalHeaderLabels(
            ["ФИО", "ID", "Вероятность посещения", "Вероятность пропуска", "Топ факторов"]
        )
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.group_table.setMinimumHeight(140)
        self.plot_hist = pg.PlotWidget(title="")
        self.plot_ranking = pg.PlotWidget(title="")
        self.plot_hist.setMinimumHeight(180)
        self.plot_ranking.setMinimumHeight(180)
        group_splitter = QSplitter(Qt.Orientation.Vertical)
        group_splitter.addWidget(self.group_table)
        group_splitter.addWidget(self.plot_hist)
        group_splitter.addWidget(self.plot_ranking)
        group_splitter.setStretchFactor(0, 2)
        group_splitter.setStretchFactor(1, 1)
        group_splitter.setStretchFactor(2, 1)
        layout.addWidget(group_splitter)

        self.student_result.setVisible(True)
        self.plot_student_factors.setVisible(True)
        self.group_table.setVisible(False)
        self.plot_hist.setVisible(False)
        self.plot_ranking.setVisible(False)
        self.radio_student.toggled.connect(lambda checked: self._switch_mode(checked))
        self._switch_mode(self.radio_student.isChecked())

    def _switch_mode(self, student_mode: bool) -> None:
        self.student_result.setVisible(student_mode)
        self.plot_student_factors.setVisible(student_mode)
        self.group_table.setVisible(not student_mode)
        self.plot_hist.setVisible(not student_mode)
        self.plot_ranking.setVisible(not student_mode)

    def _on_calc(self) -> None:
        if not self.predictor:
            self.status_label.setText("Предиктор недоступен.")
            return
        lesson_id = self.lesson_spin.value()
        lesson_date = self.date_edit.date().toString("yyyy-MM-dd")

        if self.radio_student.isChecked():
            student_id = self.student_spin.value()
            self.status_label.setText("Расчёт…")
            self.student_result.setText("")
            self._worker = PredictStudentWorker(self.predictor, student_id, lesson_id, lesson_date)
            self._thread = run_worker(
                self._worker,
                on_finished=self._on_student_result,
                on_error=self._on_error,
            )
        else:
            group = self.group_spin.value()
            self.status_label.setText("Расчёт…")
            self.group_table.setRowCount(0)
            self._worker = PredictGroupWorker(self.predictor, group, lesson_id, lesson_date)
            self._thread = run_worker(
                self._worker,
                on_finished=self._on_group_result,
                on_error=self._on_error,
            )
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: setattr(self, "_worker", None))

    def _map_feature(self, feature_name: str) -> str:
        """Возвращает человекочитаемое русское название признака."""
        return FEATURE_MAP.get(feature_name, feature_name)

    def _on_student_result(self, pred: StudentPrediction) -> None:
        self.status_label.setText("")
        lines = [
            f"Студент ID {pred.student_id}" + (f" ({pred.full_name})" if pred.full_name else ""),
            f"Вероятность посещения: {pred.attendance_probability:.2%}",
            f"Вероятность пропуска: {pred.absence_probability:.2%}",
            "Топ факторов:",
        ]
        for f in pred.top_factors:
            readable = self._map_feature(f.feature)
            lines.append(f"  • {readable}: {f.effect} (вклад {f.impact:.3f})")
        self.student_result.setText("\n".join(lines))
        plot_student_factors(self.plot_student_factors.getPlotItem(), pred.top_factors)

    def _on_group_result(self, results: List[StudentPrediction]) -> None:
        self.status_label.setText("")
        self.group_table.setRowCount(len(results))
        for i, pred in enumerate(results):
            name = pred.full_name or "—"
            factors_short = "; ".join(self._map_feature(f.feature) for f in pred.top_factors[:3])
            self.group_table.setItem(i, 0, QTableWidgetItem(name))
            self.group_table.setItem(i, 1, QTableWidgetItem(str(pred.student_id)))
            self.group_table.setItem(i, 2, QTableWidgetItem(f"{pred.attendance_probability:.2%}"))
            self.group_table.setItem(i, 3, QTableWidgetItem(f"{pred.absence_probability:.2%}"))
            self.group_table.setItem(i, 4, QTableWidgetItem(factors_short))
        probs = [r.attendance_probability for r in results]
        plot_probability_histogram(self.plot_hist.getPlotItem(), probs, "Распределение вероятностей по группе")
        plot_student_ranking(self.plot_ranking.getPlotItem(), results, top_n=20)

    def _on_error(self, msg: str) -> None:
        logger.warning("Прогноз: ошибка — %s", msg)
        self.status_label.setText(f"Ошибка: {msg}")
        self.student_result.setText("")
        self.group_table.setRowCount(0)
