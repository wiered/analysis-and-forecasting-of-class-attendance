"""
Воркеры для вызова предиктора и модели в фоновом потоке (QThread).
Результаты передаются в UI через сигналы.
"""
import logging
from typing import Optional, List

from PySide6.QtCore import QObject, Signal, QThread, Qt, QTimer

from src.model import (
    AttendancePredictor,
    StudentPrediction,
    GroupSummary,
    RescheduleEffect,
)
from src.agents import AttendanceModel, LessonPolicy

logger = logging.getLogger(__name__)


class GroupSummaryWorker(QObject):
    """Вызов get_group_summary в фоне."""
    finished = Signal(object)   # GroupSummary
    error = Signal(str)

    def __init__(self, predictor: AttendancePredictor, group: int, lesson_id: int, lesson_date: str):
        super().__init__()
        logger.debug("GroupSummaryWorker init")
        self.predictor = predictor
        self.group = group
        self.lesson_id = lesson_id
        self.lesson_date = lesson_date

    def _schedule_run(self) -> None:
        """Вызывается по started(); откладывает run() на следующий цикл event loop потока."""
        logger.debug("GroupSummaryWorker._schedule_run: планирование run()")
        QTimer.singleShot(0, self.run)

    def run(self) -> None:
        logger.debug("GroupSummaryWorker.run: начало, group=%s lesson_id=%s lesson_date=%s", self.group, self.lesson_id, self.lesson_date)
        try:
            summary = self.predictor.get_group_summary(
                self.group, self.lesson_id, self.lesson_date
            )
            logger.debug("GroupSummaryWorker.run: get_group_summary выполнен, студентов=%s", summary.total_students)
            self.finished.emit(summary)
            logger.debug("GroupSummaryWorker.run: сигнал finished испущен")
        except Exception as e:
            logger.exception("GroupSummaryWorker.run: get_group_summary failed: group=%s lesson_id=%s lesson_date=%s", self.group, self.lesson_id, self.lesson_date)
            self.error.emit(str(e))
            logger.debug("GroupSummaryWorker.run: сигнал error испущен")


class PredictGroupWorker(QObject):
    """Вызов predict_group в фоне."""
    finished = Signal(list)   # List[StudentPrediction]
    error = Signal(str)

    def __init__(self, predictor: AttendancePredictor, group: int, lesson_id: int, lesson_date: str):
        super().__init__()
        self.predictor = predictor
        self.group = group
        self.lesson_id = lesson_id
        self.lesson_date = lesson_date

    def _schedule_run(self) -> None:
        QTimer.singleShot(0, self.run)

    def run(self) -> None:
        try:
            results = self.predictor.predict_group(
                self.group, self.lesson_id, self.lesson_date, verbose=False
            )
            self.finished.emit(results)
        except Exception as e:
            logger.exception("predict_group failed: group=%s lesson_id=%s lesson_date=%s", self.group, self.lesson_id, self.lesson_date)
            self.error.emit(str(e))


class PredictStudentWorker(QObject):
    """Вызов predict_student в фоне."""
    finished = Signal(object)   # StudentPrediction
    error = Signal(str)

    def __init__(
        self,
        predictor: AttendancePredictor,
        student_id: int,
        lesson_id: int,
        lesson_date: str,
    ):
        super().__init__()
        self.predictor = predictor
        self.student_id = student_id
        self.lesson_id = lesson_id
        self.lesson_date = lesson_date

    def _schedule_run(self) -> None:
        QTimer.singleShot(0, self.run)

    def run(self) -> None:
        try:
            result = self.predictor.predict_student(
                self.student_id, self.lesson_id, self.lesson_date
            )
            self.finished.emit(result)
        except Exception as e:
            logger.exception("predict_student failed: student_id=%s lesson_id=%s lesson_date=%s", self.student_id, self.lesson_id, self.lesson_date)
            self.error.emit(str(e))


class RescheduleEffectWorker(QObject):
    """Вызов get_reschedule_effect в фоне."""
    finished = Signal(object)   # RescheduleEffect
    error = Signal(str)

    def __init__(
        self,
        predictor: AttendancePredictor,
        group: int,
        lesson_id: int,
        lesson_date: str,
        new_weekday: str,
        new_time_slot: str,
    ):
        super().__init__()
        self.predictor = predictor
        self.group = group
        self.lesson_id = lesson_id
        self.lesson_date = lesson_date
        self.new_weekday = new_weekday
        self.new_time_slot = new_time_slot

    def _schedule_run(self) -> None:
        QTimer.singleShot(0, self.run)

    def run(self) -> None:
        try:
            effect = self.predictor.get_reschedule_effect(
                self.group,
                self.lesson_id,
                self.lesson_date,
                self.new_weekday,
                self.new_time_slot,
            )
            self.finished.emit(effect)
        except Exception as e:
            logger.exception("get_reschedule_effect failed: group=%s lesson_id=%s", self.group, self.lesson_id)
            self.error.emit(str(e))


class RecommendationsWorker(QObject):
    """Запуск AttendanceModel.step() и получение policy от TeacherAgent."""
    finished = Signal(object)   # LessonPolicy
    error = Signal(str)

    def __init__(
        self,
        predictor: AttendancePredictor,
        group: int,
        lesson_id: int,
        lesson_date: str,
        reschedule_weekday: Optional[str] = None,
        reschedule_time_slot: Optional[str] = None,
    ):
        super().__init__()
        self.predictor = predictor
        self.group = group
        self.lesson_id = lesson_id
        self.lesson_date = lesson_date
        self.reschedule_weekday = reschedule_weekday or "Wednesday"
        self.reschedule_time_slot = reschedule_time_slot or "10:30"

    def _schedule_run(self) -> None:
        QTimer.singleShot(1, self.run)

    def run(self) -> None:
        try:
            model = AttendanceModel(
                predictor=self.predictor,
                lesson_id=self.lesson_id,
                lesson_date=self.lesson_date,
                group=self.group,
                reschedule_weekday=self.reschedule_weekday,
                reschedule_time_slot=self.reschedule_time_slot,
            )
            model.step()
            teacher_agents = [a for a in model.agents if hasattr(a, "policy") and a.policy is not None]
            if not teacher_agents:
                logger.warning("TeacherAgent policy not found for group=%s lesson_id=%s", self.group, self.lesson_id)
                self.error.emit("Политика преподавателя не получена")
                return
            self.finished.emit(teacher_agents[0].policy)
        except Exception as e:
            logger.exception("recommendations failed: group=%s lesson_id=%s lesson_date=%s", self.group, self.lesson_id, self.lesson_date)
            self.error.emit(str(e))


def run_worker(worker: QObject, on_finished, on_error) -> QThread:
    """Запускает воркер в отдельном потоке; подключает сигналы и стартует поток."""
    logger.debug("run_worker: создание потока, подключение сигналов")
    thread = QThread()
    worker.moveToThread(thread)

    schedule_run = getattr(worker, "_schedule_run", worker.run)
    thread.started.connect(schedule_run, Qt.ConnectionType.QueuedConnection)

    worker.finished.connect(on_finished)
    worker.error.connect(on_error)

    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)

    worker.finished.connect(worker.deleteLater)
    worker.error.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread
