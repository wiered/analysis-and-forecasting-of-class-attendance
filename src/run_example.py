"""
Минимальный пример запуска Mesa-модели с агентами студента, преподавателя и деканата.
Требует доступную БД и файлы модели (attendance_model.cbm, feature_names.pkl).
Конфиг БД берётся из .env (см. src.settings).

Запуск из корня репозитория:
  python -m src.run_example
"""
from src.model import AttendancePredictor
from src.settings import db_config
from src.agents import (
    AttendanceModel,
    StudentAgent,
    TeacherAgent,
    DeaneryAgent,
)


def main() -> None:
    try:
        predictor = AttendancePredictor(db_config=db_config)
    except Exception as e:
        print(
            "Не удалось создать AttendancePredictor (нужны БД и файлы "
            "attendance_model.cbm, feature_names.pkl):",
            e,
        )
        return
    model = AttendanceModel(
        predictor=predictor,
        lesson_id=2,
        lesson_date="2026-03-04",
        group=1,
        student_id=1,
        reschedule_weekday="Wednesday",
        reschedule_time_slot="10:30",
    )
    model.step()

    for agent in model.agents:
        if isinstance(agent, StudentAgent):
            print("StudentAgent prediction:", agent.prediction)
            print("StudentAgent decision:", agent.decision)
        elif isinstance(agent, TeacherAgent):
            print("TeacherAgent policy:", agent.policy)
            if agent.policy and agent.policy.tactics:
                print("  Тактики:", [t.name for t in agent.policy.tactics])
        elif isinstance(agent, DeaneryAgent):
            print("DeaneryAgent reschedule_effect:", agent.reschedule_effect)
            print("DeaneryAgent decision:", agent.decision)

    student_agents = [a for a in model.agents if isinstance(a, StudentAgent)]
    teacher_agents = [a for a in model.agents if isinstance(a, TeacherAgent)]
    deanery_agents = [a for a in model.agents if isinstance(a, DeaneryAgent)]
    assert len(teacher_agents) == 1 and teacher_agents[0].policy is not None
    assert len(deanery_agents) == 1 and deanery_agents[0].reschedule_effect is not None
    assert len(deanery_agents) == 1 and deanery_agents[0].decision is not None
    if student_agents:
        assert student_agents[0].prediction is not None
        assert student_agents[0].decision is not None
    print("OK: все агенты заполнили выходы после step().")


if __name__ == "__main__":
    main()
