"""Виджет рекомендаций — политика преподавателя (LessonPolicy)."""
import logging
from typing import Optional

from src.model import AttendancePredictor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QDateEdit,
    QLabel,
    QCheckBox,
    QGroupBox,
    QFrame,
    QScrollArea,
)
from PySide6.QtCore import QDate

from src.agents import LessonPolicy

from .workers import RecommendationsWorker, run_worker

logger = logging.getLogger(__name__)


class RecommendationsWidget(QWidget):
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

        btn = QPushButton("Получить рекомендации")
        btn.clicked.connect(self._on_load)
        layout.addWidget(btn)

        self.status_label = QLabel("Введите параметры и нажмите кнопку.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.policy_frame = QFrame()
        self.policy_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        policy_layout = QVBoxLayout(self.policy_frame)
        self.use_interactive_cb = QCheckBox("Использовать интерактивные элементы")
        self.use_interactive_cb.setEnabled(False)
        self.use_quizzes_cb = QCheckBox("Проводить микроквизы")
        self.use_quizzes_cb.setEnabled(False)
        self.strengthen_control_cb = QCheckBox("Усилить контроль присутствия")
        self.strengthen_control_cb.setEnabled(False)
        policy_layout.addWidget(self.use_interactive_cb)
        policy_layout.addWidget(self.use_quizzes_cb)
        policy_layout.addWidget(self.strengthen_control_cb)
        self.recommendations_label = QLabel("")
        self.recommendations_label.setWordWrap(True)
        policy_layout.addWidget(QLabel("Рекомендации:"))
        policy_layout.addWidget(self.recommendations_label)
        self.policy_frame.setVisible(False)
        layout.addWidget(self.policy_frame)

        layout.addStretch()

    def _on_load(self) -> None:
        if not self.predictor:
            self.status_label.setText("Предиктор недоступен.")
            return
        group = self.group_spin.value()
        lesson_id = self.lesson_spin.value()
        lesson_date = self.date_edit.date().toString("yyyy-MM-dd")

        self.status_label.setText("Загрузка…")
        self.policy_frame.setVisible(False)

        self._worker = RecommendationsWorker(self.predictor, group, lesson_id, lesson_date)
        self._thread = run_worker(
            self._worker,
            on_finished=self._on_policy,
            on_error=self._on_error,
        )
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: setattr(self, "_worker", None))

    def _on_policy(self, policy: LessonPolicy) -> None:
        self.status_label.setText("")
        self.use_interactive_cb.setChecked(policy.use_interactive)
        self.use_quizzes_cb.setChecked(policy.use_quizzes)
        self.strengthen_control_cb.setChecked(policy.strengthen_attendance_control)
        self.recommendations_label.setText("\n".join(f"• {r}" for r in policy.recommendations) or "—")
        self.policy_frame.setVisible(True)

    def _on_error(self, msg: str) -> None:
        logger.warning("Рекомендации: ошибка — %s", msg)
        self.status_label.setText(f"Ошибка: {msg}")
        self.policy_frame.setVisible(False)
