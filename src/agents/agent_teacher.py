from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
from mesa import Agent

if TYPE_CHECKING:
    from .attendance_model import AttendanceModel


@dataclass
class LessonPolicy:
    """Политика проведения занятия (рекомендации на основе сводки по группе)."""
    use_interactive: bool
    use_quizzes: bool
    strengthen_attendance_control: bool
    recommendations: List[str]


class TeacherAgent(Agent):
    """
    Агент преподавателя: использует только существующий API (get_group_summary).
    Формирует политику по risk_percentage и top_group_factors.
    """

    def __init__(self, model: "AttendanceModel"):
        super().__init__(model)
        self.policy: Optional[LessonPolicy] = None

    def step(self) -> None:
        model = self.model
        summary = model.predictor.get_group_summary(
            model.group,
            model.lesson_id,
            model.lesson_date,
        )
        recommendations: List[str] = []
        use_interactive = summary.risk_percentage > 30
        use_quizzes = summary.risk_percentage > 20
        strengthen_attendance_control = summary.risk_percentage > 40

        if summary.risk_percentage > 40:
            recommendations.append("Усилить контроль присутствия")
        if summary.risk_percentage > 30:
            recommendations.append("Добавить интерактивные элементы")
        if summary.risk_percentage > 20:
            recommendations.append("Провести микроквиз для вовлечённости")
        for gf in summary.top_group_factors[:2]:
            recommendations.append(f"Учесть фактор риска: {gf.feature}")

        self.policy = LessonPolicy(
            use_interactive=use_interactive,
            use_quizzes=use_quizzes,
            strengthen_attendance_control=strengthen_attendance_control,
            recommendations=recommendations,
        )
