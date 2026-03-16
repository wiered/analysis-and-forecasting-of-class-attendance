"""
Воркеры для вызова предиктора и модели в фоновом потоке (QThread).
Результаты передаются в UI через сигналы.
"""
import logging
from datetime import date, timedelta
from typing import Optional, List

from PySide6.QtCore import QObject, Signal, QThread, Qt, QTimer

from src.model import (
    AttendancePredictor,
    StudentPrediction,
    GroupSummary,
    RescheduleEffect,
    BestSlotResult,
)
from src.agents import (
    AttendanceModel,
    LessonPolicy,
    StudentAgent,
    TeacherAgent,
    DeaneryAgent,
)
from src.utils import map_factors

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


class BestSlotWorker(QObject):
    """Вызов find_best_reschedule_slot в фоне."""
    finished = Signal(object)   # BestSlotResult
    error = Signal(str)

    def __init__(
        self,
        predictor: AttendancePredictor,
        group: int,
        lesson_id: int,
        lesson_date: str,
    ):
        super().__init__()
        self.predictor = predictor
        self.group = group
        self.lesson_id = lesson_id
        self.lesson_date = lesson_date

    def _schedule_run(self) -> None:
        QTimer.singleShot(0, self.run)

    def run(self) -> None:
        try:
            result = self.predictor.find_best_reschedule_slot(
                self.group,
                self.lesson_id,
                self.lesson_date,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.exception(
                "find_best_reschedule_slot failed: group=%s lesson_id=%s",
                self.group, self.lesson_id,
            )
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


def _next_weekdays(start: date, count: int = 5) -> List[date]:
    """Следующие count рабочих дней (Пн–Пт) начиная с start."""
    out: List[date] = []
    d = start
    while len(out) < count:
        if d.weekday() < 5:  # 0=Mon .. 4=Fri
            out.append(d)
        d += timedelta(days=1)
    return out


WEEKDAY_NAMES = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")

# Полные названия дней для лога переноса (англ. из БД → рус.)
WEEKDAY_FULL_RU = {
    "Monday": "Понедельник",
    "Tuesday": "Вторник",
    "Wednesday": "Среда",
    "Thursday": "Четверг",
    "Friday": "Пятница",
    "Saturday": "Суббота",
}


class SimulationWorker(QObject):
    """
    Симуляция 1 недели (5 рабочих дней): студенты решают идти/не идти,
    преподаватель выбирает стратегии, деканат принимает решения по расписанию и уведомлениям.
    Логи формируются через Mesa-агентов и предиктор.
    """
    finished = Signal(str)   # полный текст лога
    error = Signal(str)

    def __init__(
        self,
        predictor: AttendancePredictor,
        group: int,
        lesson_id: int,
        start_date: date,
        reschedule_weekday: Optional[str] = None,
        reschedule_time_slot: Optional[str] = None,
    ):
        super().__init__()
        self.predictor = predictor
        self.group = group
        self.lesson_id = lesson_id
        self.start_date = start_date
        # None, None — деканат подбирает лучший слот (find_best_reschedule_slot)
        self.reschedule_weekday = reschedule_weekday
        self.reschedule_time_slot = reschedule_time_slot

    def _schedule_run(self) -> None:
        QTimer.singleShot(0, self.run)

    def run(self) -> None:
        try:
            days = _next_weekdays(self.start_date, 5)
            lines: List[str] = []
            lines.append("=== СИМУЛЯЦИЯ НЕДЕЛИ (5 рабочих дней) ===")
            lines.append(f"Группа: {self.group}, Занятие ID: {self.lesson_id}")
            lines.append("")

            for day_num, d in enumerate(days, 1):
                date_str = d.strftime("%Y-%m-%d")
                wd = WEEKDAY_NAMES[d.weekday()]
                lines.append("")
                lines.append(f"——— День {day_num} ({date_str}, {wd}) ———")

                # Сначала шаг модели: преподаватель и деканат принимают решения
                try:
                    model = AttendanceModel(
                        predictor=self.predictor,
                        lesson_id=self.lesson_id,
                        lesson_date=date_str,
                        group=self.group,
                        reschedule_weekday=self.reschedule_weekday,
                        reschedule_time_slot=self.reschedule_time_slot,
                    )
                    model.step()
                except Exception as e:
                    lines.append(f"  [Ошибка Mesa: {e}]")
                    self.finished.emit("\n".join(lines))
                    return

                teacher_agents = [a for a in model.agents if isinstance(a, TeacherAgent)]
                if teacher_agents and teacher_agents[0].policy:
                    pol = teacher_agents[0].policy
                    lines.append(f"  [Преподаватель] Политика: интерактив={pol.use_interactive}, квизы={pol.use_quizzes}, контроль={pol.strengthen_attendance_control}.")
                    for t in pol.tactics[:3]:
                        lines.append(f"    Тактика: {t.name} ({t.priority}) — {t.description}")
                    if pol.recommendations:
                        lines.append(f"    Рекомендации: {'; '.join(pol.recommendations[:3])}.")

                deanery_agents = [a for a in model.agents if isinstance(a, DeaneryAgent)]
                if deanery_agents and deanery_agents[0].decision:
                    dec = deanery_agents[0].decision
                    deanery_agent = deanery_agents[0]
                    is_transfer = dec.schedule_decision.startswith("Рекомендуется перенос")

                    if is_transfer:
                        # Деканат осуществляет перенос; формат: перенос: Среда 8:30 → Понедельник 10:30
                        current_schedule = self.predictor.get_lesson_schedule(self.lesson_id)
                        if current_schedule:
                            from_ru = WEEKDAY_FULL_RU.get(current_schedule[0], current_schedule[0])
                            from_time = current_schedule[1].lstrip("0") if current_schedule[1].startswith("0") else current_schedule[1]
                        else:
                            from_ru = "?"
                            from_time = "?"
                        to_ru = WEEKDAY_FULL_RU.get(deanery_agent.new_weekday, deanery_agent.new_weekday)
                        to_time = (deanery_agent.new_time_slot.lstrip("0") if deanery_agent.new_time_slot.startswith("0") else deanery_agent.new_time_slot)
                        # Убираем из текста решения первую фразу "Рекомендуется перенос: date → wd ts. "
                        idx = dec.schedule_decision.find(". ", 0)
                        reasoning = dec.schedule_decision[idx + 2 :].strip() if idx >= 0 else dec.schedule_decision
                        lines.append(f"  [Деканат] Расписание: перенос: {from_ru} {from_time} → {to_ru} {to_time}. {reasoning}")
                        # Студенты получают уведомление и пересчитывают решения по новому расписанию
                        try:
                            predictions = self.predictor.predict_group(
                                self.group, self.lesson_id, date_str,
                                schedule_override={"weekday": deanery_agent.new_weekday, "time_slot": deanery_agent.new_time_slot},
                                verbose=False,
                            )
                        except Exception as e:
                            lines.append(f"  [Ошибка прогноза по группе: {e}]")
                            predictions = []
                        lines.append(f"  [Деканат] Сообщение {len(predictions)} студентам о переносе пары")
                        for p in predictions:
                            attend = p.attendance_probability >= 0.5
                            name = (p.full_name or f"Студент {p.student_id}").strip()
                            reasons = map_factors(p.top_factors) if p.top_factors else "Нет данных."
                            decision = "идёт" if attend else "пропускает"
                            lines.append(f"  [Студент] {name}: решение — {decision}. {reasons}")
                        if not predictions:
                            lines.append("  [Студент] Нет данных по группе.")
                    else:
                        lines.append(f"  [Деканат] Расписание: {dec.schedule_decision}")
                        try:
                            predictions = self.predictor.predict_group(
                                self.group, self.lesson_id, date_str, verbose=False
                            )
                        except Exception as e:
                            lines.append(f"  [Ошибка прогноза по группе: {e}]")
                            predictions = []
                        for p in predictions:
                            attend = p.attendance_probability >= 0.5
                            name = (p.full_name or f"Студент {p.student_id}").strip()
                            reasons = map_factors(p.top_factors) if p.top_factors else "Нет данных."
                            decision = "идёт" if attend else "пропускает"
                            lines.append(f"  [Студент] {name}: решение — {decision}. {reasons}")
                        if not predictions:
                            lines.append("  [Студент] Нет данных по группе.")

                    lines.append(f"  [Деканат] Аудитория: {dec.classroom_recommendation}")
                    for act in dec.notification_actions:
                        lines.append(f"  [Деканат] Уведомления: {act}")

            log_text = "\n".join(lines)
            self.finished.emit(log_text)
        except Exception as e:
            logger.exception("SimulationWorker.run failed: group=%s lesson_id=%s", self.group, self.lesson_id)
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
