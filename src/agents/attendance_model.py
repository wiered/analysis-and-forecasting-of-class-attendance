from typing import Optional

from mesa import Model

from ..model import AttendancePredictor

from .agent_student import StudentAgent
from .agent_teacher import TeacherAgent
from .agent_deanery import DeaneryAgent


class AttendanceModel(Model):
    """
    Mesa-модель посещаемости: хранит предиктор и контекст занятия,
    управляет агентами студента, преподавателя и деканата.
    """

    def __init__(
        self,
        predictor: AttendancePredictor,
        lesson_id: int,
        lesson_date: str,
        group: int,
        student_id: Optional[int] = None,
        reschedule_weekday: Optional[str] = None,
        reschedule_time_slot: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.predictor = predictor
        self.lesson_id = lesson_id
        self.lesson_date = lesson_date
        self.group = group
        self._student_id = student_id
        self._reschedule_weekday = reschedule_weekday or "Wednesday"
        self._reschedule_time_slot = reschedule_time_slot or "10:30"

        if student_id is not None:
            StudentAgent(self, student_id=student_id)
        TeacherAgent(self)
        DeaneryAgent(
            self,
            new_weekday=self._reschedule_weekday,
            new_time_slot=self._reschedule_time_slot,
        )

    def step(self) -> None:
        """Шаг модели: активирует всех агентов (их метод step)."""
        self.agents.do("step")
