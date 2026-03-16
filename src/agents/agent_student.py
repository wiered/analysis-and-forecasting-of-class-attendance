from dataclasses import dataclass
from mesa import Agent
from typing import Optional, TYPE_CHECKING

from src.utils import map_factors

if TYPE_CHECKING:
    from .attendance_model import AttendanceModel


@dataclass
class StudentAttendanceDecision:
    """
    Решение студента о посещении занятия с обоснованием.

    Attributes:
        attend (bool): Идти на занятие (True) или пропустить (False).
        justification (str): Текстовое обоснование решения.
    """
    attend: bool
    justification: str


class StudentAgent(Agent):
    """
    Агент студента: формирует решение о посещении занятия с текстовым обоснованием.
    В step() вызывает предиктор, сохраняет результат в self.prediction
    и формирует self.decision (посещать/не посещать + обоснование).
    """

    def __init__(
        self,
        model: "AttendanceModel",
        student_id: int,
    ):
        super().__init__(model)
        self.student_id = student_id
        self.prediction = None
        self.decision: Optional[StudentAttendanceDecision] = None

    def _build_justification(self) -> str:
        """Формирует текстовое обоснование на основе прогноза и факторов (маппер из utils)."""
        p = self.prediction
        if p is None:
            return "Нет данных для принятия решения."

        prob = p.attendance_probability
        parts = [f"Вероятность посещения {prob:.0%}, пропуска {p.absence_probability:.0%}."]
        if p.top_factors:
            parts.append(" " + map_factors(p.top_factors))
        if prob >= 0.5:
            parts.append(" Итог: решение посещать занятие.")
        else:
            parts.append(" Итог: высокий риск пропуска; решение не посещать.")
        return "".join(parts)

    def step(self) -> None:
        model = self.model
        self.prediction = model.predictor.predict_student(
            self.student_id,
            model.lesson_id,
            model.lesson_date,
        )
        attend = self.prediction.attendance_probability >= 0.5
        self.decision = StudentAttendanceDecision(
            attend=attend,
            justification=self._build_justification(),
        )
