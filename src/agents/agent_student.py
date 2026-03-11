from mesa import Agent
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .attendance_model import AttendanceModel


class StudentAgent(Agent):
    """
    Агент студента: обёртка над predict_student.
    В step() вызывает предиктор и сохраняет результат в self.prediction.
    """

    def __init__(
        self,
        model: "AttendanceModel",
        student_id: int,
    ):
        super().__init__(model)
        self.student_id = student_id
        self.prediction = None

    def step(self) -> None:
        model = self.model
        self.prediction = model.predictor.predict_student(
            self.student_id,
            model.lesson_id,
            model.lesson_date,
        )
