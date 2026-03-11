from mesa import Agent
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .attendance_model import AttendanceModel


class DeaneryAgent(Agent):
    """
    Агент деканата: вычисляет ожидаемый эффект от переноса занятия.
    В step() вызывает get_reschedule_effect и сохраняет результат в self.reschedule_effect.
    """

    def __init__(
        self,
        model: "AttendanceModel",
        new_weekday: str,
        new_time_slot: str,
    ):
        super().__init__(model)
        self.new_weekday = new_weekday
        self.new_time_slot = new_time_slot
        self.reschedule_effect = None

    def step(self) -> None:
        model = self.model
        self.reschedule_effect = model.predictor.get_reschedule_effect(
            model.group,
            model.lesson_id,
            model.lesson_date,
            self.new_weekday,
            self.new_time_slot,
        )
