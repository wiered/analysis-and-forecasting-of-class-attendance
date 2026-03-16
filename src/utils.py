# ======================================================
# ================= EXPLANATION MAPPING =================
# ======================================================

from typing import List
from src.model.predict import FactorImpact
from src.model.predict import GroupSummary

def map_factors(top_factors: List[FactorImpact]) -> str:
    """
    Формирует текстовое объяснение решения модели
    на основе топ факторов.
    Args:
        top_factors (List[FactorImpact]): список факторов влияния
    Returns:
        str: человекочитаемое объяснение
    """
    feature_map = {
        "gpa": "успеваемость",
        "commute_time": "долгая дорога до университета",
        "works": "наличие работы",
        "dorm": "проживание в общежитии",
        "motivation": "мотивация",
        "discipline": "дисциплина",
        "anxiety": "тревожность",
        "burnout": "выгорание",
        "extraversion": "социальная активность",
        "deadlines_count": "дедлайны",
        "avg_attendance_last4": "посещаемость за последние недели",
        "prev_absence": "предыдущий пропуск занятия",
        "weekday": "день недели",
        "time_slot": "время занятия",
        "duration": "длительность занятия",
        "early_class": "ранняя пара",
        "friends_mean_attendance": "посещаемость друзей",
        "fairness": "восприятие справедливости преподавателя",
        "clarity": "понятность объяснения преподавателя",
        "sympathy": "симпатия к преподавателю",
        "fear": "страх перед преподавателем",
        "usefulness": "воспринимаемая полезность предмета"
    }
    explanations = []
    for factor in top_factors:
        readable_name = feature_map.get(factor.feature, factor.feature)
        if factor.effect == "увеличивает риск пропуска":
            phrase = f"{readable_name} увеличивает вероятность пропуска на " + str(factor.impact) + "%"
        else:
            phrase = f"{readable_name} повышает вероятность посещения на " + str(factor.impact) + "%"
        explanations.append(phrase)
    return "Основные причины прогноза: " + "; ".join(explanations) + "."

def map_group_factors(group_summary: GroupSummary) -> str:
    """
    Формирует текстовое объяснение для сводки по группе.
    Args:
        group_summary (GroupSummary): сводка по группе
    Returns:
        str: человекочитаемое объяснение
    """
    feature_map = {
        "gpa": "успеваемость",
        "commute_time": "время на дорогу",
        "works": "трудоустройство",
        "dorm": "проживание в общежитии",
        "motivation": "мотивация",
        "discipline": "дисциплина",
        "anxiety": "тревожность",
        "burnout": "выгорание",
        "extraversion": "социальная активность",
        "deadlines_count": "учебная нагрузка",
        "avg_attendance_last4": "прошлая посещаемость",
        "prev_absence": "предыдущие пропуски",
        "weekday": "день недели",
        "time_slot": "время проведения",
         "duration": "длительность пары",
        "early_class": "ранние занятия",
        "friends_mean_attendance": "влияние окружения",
        "fairness": "отношение к справедливости преподавателя",
        "clarity": "понятность материала",
        "sympathy": "отношение к преподавателю",
        "fear": "страх перед преподавателем",
        "usefulness": "полезность предмета"
    }

    # Базовая статистика
    stats = [
        f"Группа {group_summary.group}",
        f"всего студентов: {group_summary.total_students}",
        f"средняя вероятность посещения: {group_summary.avg_attendance_probability:.1%}",
        f"студентов в зоне риска: {group_summary.students_at_risk} ({group_summary.risk_percentage:.1f}%)"
    ]

    base_info = "Статистика: " + ", ".join(stats) + "."

    # Анализ факторов
    if not group_summary.top_group_factors:
        return base_info + " Значимых факторов риска не выявлено."

    factor_parts = []
    for factor in group_summary.top_group_factors[:3]:  # Берем топ-3 для краткости
        readable_name = feature_map.get(factor.feature, factor.feature)
        students_percent = (factor.students_affected / group_summary.total_students) * 100

        factor_parts.append(
            f"{readable_name} влияет на {factor.students_affected} "
            f"студентов ({students_percent:.0f}%), "
            f"средний вклад {factor.avg_impact:.3f}"
        )

    factors_text = "Ключевые факторы: " + "; ".join(factor_parts) + "."

    # Оценка ситуации
    if group_summary.risk_percentage < 20:
        assessment = "Ситуация благоприятная, риск пропусков низкий."
    elif group_summary.risk_percentage < 40:
        assessment = "Ситуация требует внимания, повышенный риск пропусков."
    else:
        assessment = "Критическая ситуация, высокий риск пропусков в группе."

    return f"{base_info} {factors_text} {assessment}"

