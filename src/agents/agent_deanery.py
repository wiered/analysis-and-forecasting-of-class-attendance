from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
from mesa import Agent

from src.utils import map_group_factors

if TYPE_CHECKING:
    from .attendance_model import AttendanceModel


@dataclass
class DeaneryDecision:
    """
    Управленческие решения деканата на основе агрегированных данных.

    Attributes:
        schedule_decision (str): Рекомендация по расписанию (оставить/перенести и обоснование).
        classroom_recommendation (str): Рекомендация по аудитории (вместимость, тип).
        notification_actions (List[str]): Действия по адресным уведомлениям.
        group_summary_text (str): Человекочитаемое описание сводки по группе (map_group_factors).
    """
    schedule_decision: str
    classroom_recommendation: str
    notification_actions: List[str]
    group_summary_text: str = ""


class DeaneryAgent(Agent):
    """
    Агент деканата: принимает управленческие решения по расписанию, аудиториям
    и адресным уведомлениям на основе агрегированных данных (сводка по группе,
    эффект переноса).
    """

    def __init__(
        self,
        model: "AttendanceModel",
        new_weekday: Optional[str] = None,
        new_time_slot: Optional[str] = None,
    ):
        super().__init__(model)
        self.new_weekday = new_weekday
        self.new_time_slot = new_time_slot
        self.reschedule_effect = None
        self.decision: Optional[DeaneryDecision] = None

    def step(self) -> None:
        model = self.model
        summary = model.predictor.get_group_summary(
            model.group,
            model.lesson_id,
            model.lesson_date,
        )

        if self.new_weekday is not None and self.new_time_slot is not None:
            self.reschedule_effect = model.predictor.get_reschedule_effect(
                model.group,
                model.lesson_id,
                model.lesson_date,
                self.new_weekday,
                self.new_time_slot,
            )
        else:
            best = model.predictor.find_best_reschedule_slot(
                model.group,
                model.lesson_id,
                model.lesson_date,
            )
            self.new_weekday = best.best_weekday
            self.new_time_slot = best.best_time_slot
            self.reschedule_effect = best.reschedule_effect

        eff = self.reschedule_effect
        current_slot_text = ""
        if eff.delta <= 0.05:
            current = model.predictor.get_lesson_schedule(model.lesson_id)
            if current:
                current_slot_text = f" (текущий слот: {current[0]} {current[1]})"

        # Пороги: дельта посещаемости и доля в зоне риска
        DELTA_THRESHOLD = 0.05
        RISK_HIGH_PCT = 40.0
        RISK_IMPROVEMENT_PP = 5.0  # снижение доли риска в п.п. для учёта при малой дельте
        risk_reduction = eff.risk_pct_before - eff.risk_pct_after
        risk_increase = eff.risk_pct_after - eff.risk_pct_before

        # risk_pct_* уже в шкале 0–100 (проценты), форматируем как число + "%"
        risk_before_str = f"{eff.risk_pct_before:.0f}%"
        risk_after_str = f"{eff.risk_pct_after:.0f}%"

        if eff.delta > DELTA_THRESHOLD:
            schedule_decision = (
                f"Рекомендуется перенос: {model.lesson_date} → {self.new_weekday} {self.new_time_slot}. "
                f"Ожидаемый рост посещаемости на {eff.delta:.0%}, доля в зоне риска снизится с "
                f"{risk_before_str} до {risk_after_str}."
            )
        elif eff.delta < -DELTA_THRESHOLD:
            schedule_decision = (
                f"Перенос не рекомендуется: при смене на {self.new_weekday} {self.new_time_slot} "
                f"посещаемость ожидаемо снизится на {abs(eff.delta):.0%}. Оставить текущее расписание."
            )
        elif eff.risk_pct_before >= RISK_HIGH_PCT and risk_reduction >= RISK_IMPROVEMENT_PP:
            schedule_decision = (
                f"Рекомендуется перенос: {model.lesson_date} → {self.new_weekday} {self.new_time_slot}. "
                f"При текущем расписании доля студентов в зоне риска высокая ({risk_before_str}); "
                f"перенос снизит её до {risk_after_str} (Δ посещаемости {eff.delta:+.0%})."
            )
        elif risk_increase >= RISK_IMPROVEMENT_PP:
            schedule_decision = (
                f"Перенос не рекомендуется: при смене на {self.new_weekday} {self.new_time_slot} "
                f"доля в зоне риска вырастет с {risk_before_str} до {risk_after_str}. "
                f"Оставить текущее расписание."
            )
        else:
            schedule_decision = (
                f"Существенного эффекта от переноса на {self.new_weekday} {self.new_time_slot} не ожидается "
                f"(Δ ≈ {eff.delta:.0%}, доля в зоне риска: {risk_before_str} → {risk_after_str}). "
                f"Расписание можно оставить без изменений{current_slot_text}."
            )

        # При переносе явка и уведомления считаем по расписанию после переноса
        did_transfer = schedule_decision.startswith("Рекомендуется перенос")
        if did_transfer:
            avg_att = eff.avg_attendance_after
            students_at_risk = max(0, round(summary.total_students * eff.risk_pct_after / 100))
            risk_pct = eff.risk_pct_after
        else:
            avg_att = summary.avg_attendance_probability
            students_at_risk = summary.students_at_risk
            risk_pct = summary.risk_percentage

        # Рекомендация по аудитории (ожидаемое число пришедших)
        expected_attend = int(summary.total_students * avg_att + 0.5)
        if summary.total_students <= 0:
            classroom_recommendation = "Нет данных по группе; выбрать аудиторию по списку группы."
        elif expected_attend <= 0:
            classroom_recommendation = (
                f"Ожидаемая явка очень низкая ({summary.total_students} в группе). "
                "Занятие в малой аудитории или общий сбор с другими группами."
            )
        else:
            margin = max(2, expected_attend // 5)
            classroom_recommendation = (
                f"Ожидаемая явка ≈ {expected_attend} из {summary.total_students}. "
                f"Рекомендуется аудитория на {expected_attend + margin}+ мест."
            )

        # Адресные уведомления
        notification_actions: List[str] = []
        if students_at_risk > 0:
            notification_actions.append(
                f"Направить напоминание о занятии {students_at_risk} студентам в зоне риска пропуска."
            )
        if risk_pct > 40:
            notification_actions.append(
                "Разослать в чат группы краткое напоминание о дате, времени и месте занятия."
            )
        if not notification_actions:
            notification_actions.append("Дополнительные уведомления не требуются.")

        self.decision = DeaneryDecision(
            schedule_decision=schedule_decision,
            classroom_recommendation=classroom_recommendation,
            notification_actions=notification_actions,
            group_summary_text=map_group_factors(summary),
        )
