from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
from mesa import Agent

from src.utils import map_group_factors, FEATURE_MAP

if TYPE_CHECKING:
    from .attendance_model import AttendanceModel


@dataclass
class PedagogicalTactic:
    """Одна педагогическая тактика для повышения посещаемости."""
    name: str
    description: str
    priority: str  # "высокий" | "средний" | "низкий"


@dataclass
class LessonPolicy:
    """Политика проведения занятия (рекомендации на основе сводки по группе)."""
    use_interactive: bool
    use_quizzes: bool
    strengthen_attendance_control: bool
    recommendations: List[str]
    tactics: List[PedagogicalTactic] = field(default_factory=list)
    group_summary_text: str = ""  # человекочитаемое описание по map_group_factors


class TeacherAgent(Agent):
    """
    Агент преподавателя: выбирает педагогические тактики для повышения посещаемости
    на основе прогноза по группе (get_group_summary). Формирует политику занятия
    и список тактик по risk_percentage и top_group_factors.
    """

    def __init__(self, model: "AttendanceModel"):
        super().__init__(model)
        self.policy: Optional[LessonPolicy] = None

    def _select_tactics(self, summary) -> List[PedagogicalTactic]:
        """Выбор тактик по проценту риска и топ-факторам группы."""
        tactics: List[PedagogicalTactic] = []
        r = summary.risk_percentage

        if r > 50:
            tactics.append(PedagogicalTactic(
                name="Жёсткий контроль присутствия",
                description="Фиксация присутствия в начале и конце пары, связь пропусков с аттестацией.",
                priority="высокий",
            ))
        if r > 40:
            tactics.append(PedagogicalTactic(
                name="Усиление контроля и напоминания",
                description="Усилить контроль присутствия; напомнить о важности занятия в чате группы.",
                priority="высокий",
            ))
        if r > 30:
            tactics.append(PedagogicalTactic(
                name="Интерактивные элементы",
                description="Добавить опросы, обсуждения в парах, короткие задания в течение пары.",
                priority="средний",
            ))
        if r > 20:
            tactics.append(PedagogicalTactic(
                name="Микроквизы и вовлечённость",
                description="Провести микроквиз или мини-задание для повышения вовлечённости.",
                priority="средний",
            ))
        if r > 10:
            tactics.append(PedagogicalTactic(
                name="Поддержка мотивации",
                description="Кратко обозначить ценность темы и связь с экзаменом/практикой.",
                priority="низкий",
            ))

        # Тактики по факторам риска группы
        factor_hints = {
            "weekday": ("Учёт дня недели", "По возможности назначать консультации или повтор в день с лучшей посещаемостью."),
            "time_slot": ("Учёт времени", "Учитывать типичную загрузку в это время; при необходимости обсудить с деканатом слот."),
            "distance": ("Удалённость/логистика", "Учитывать логистику; дать чёткую навигацию до аудитории."),
            "subject": ("Специфика предмета", "Подчеркнуть связь с экзаменом и практикой по предмету."),
        }
        for gf in summary.top_group_factors[:3]:
            for key, (name, desc) in factor_hints.items():
                if key in (gf.feature or "").lower():
                    tactics.append(PedagogicalTactic(
                        name=name,
                        description=desc,
                        priority="средний",
                    ))
                    break

        return tactics

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
            readable = FEATURE_MAP.get(gf.feature, gf.feature)
            recommendations.append(f"Учесть фактор риска: {readable}")

        tactics = self._select_tactics(summary)
        self.policy = LessonPolicy(
            use_interactive=use_interactive,
            use_quizzes=use_quizzes,
            strengthen_attendance_control=strengthen_attendance_control,
            recommendations=recommendations,
            tactics=tactics,
            group_summary_text=map_group_factors(summary),
        )
