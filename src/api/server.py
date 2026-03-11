"""
HTTP API для взаимодействия агентов с предиктором и моделью.
Эндпоинты — обёртки над AttendancePredictor и AttendanceModel.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from src.model import AttendancePredictor
from src.agents import AttendanceModel


def create_app(predictor: Optional[AttendancePredictor] = None) -> FastAPI:
    app = FastAPI(title="Attendance Decision Support API")

    def require_predictor() -> AttendancePredictor:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Predictor unavailable")
        return predictor

    @app.get("/group_summary")
    def group_summary(
        group: int = Query(...),
        lesson_id: int = Query(...),
        lesson_date: str = Query(...),
    ):
        p = require_predictor()
        summary = p.get_group_summary(group, lesson_id, lesson_date)
        return {
            "group": summary.group,
            "lesson_id": summary.lesson_id,
            "lesson_date": summary.lesson_date,
            "total_students": summary.total_students,
            "avg_attendance_probability": summary.avg_attendance_probability,
            "students_at_risk": summary.students_at_risk,
            "risk_percentage": summary.risk_percentage,
            "top_group_factors": [
                {"feature": gf.feature, "students_affected": gf.students_affected, "avg_impact": gf.avg_impact}
                for gf in summary.top_group_factors
            ],
        }

    @app.get("/predict_student")
    def predict_student(
        student_id: int = Query(...),
        lesson_id: int = Query(...),
        lesson_date: str = Query(...),
    ):
        p = require_predictor()
        pred = p.predict_student(student_id, lesson_id, lesson_date)
        return {
            "student_id": pred.student_id,
            "attendance_probability": pred.attendance_probability,
            "absence_probability": pred.absence_probability,
            "full_name": pred.full_name,
            "top_factors": [
                {"feature": f.feature, "impact": f.impact, "effect": f.effect}
                for f in pred.top_factors
            ],
        }

    @app.get("/predict_group")
    def predict_group(
        group: int = Query(...),
        lesson_id: int = Query(...),
        lesson_date: str = Query(...),
    ):
        p = require_predictor()
        results = p.predict_group(group, lesson_id, lesson_date, verbose=False)
        return [
            {
                "student_id": r.student_id,
                "attendance_probability": r.attendance_probability,
                "absence_probability": r.absence_probability,
                "full_name": r.full_name,
                "top_factors": [{"feature": f.feature, "impact": f.impact, "effect": f.effect} for f in r.top_factors],
            }
            for r in results
        ]

    @app.get("/reschedule_effect")
    def reschedule_effect(
        group: int = Query(...),
        lesson_id: int = Query(...),
        lesson_date: str = Query(...),
        new_weekday: str = Query(...),
        new_time_slot: str = Query(...),
    ):
        p = require_predictor()
        effect = p.get_reschedule_effect(group, lesson_id, lesson_date, new_weekday, new_time_slot)
        return {
            "avg_attendance_before": effect.avg_attendance_before,
            "avg_attendance_after": effect.avg_attendance_after,
            "delta": effect.delta,
            "risk_pct_before": effect.risk_pct_before,
            "risk_pct_after": effect.risk_pct_after,
        }

    @app.get("/recommendations")
    def recommendations(
        group: int = Query(...),
        lesson_id: int = Query(...),
        lesson_date: str = Query(...),
        reschedule_weekday: str = Query("Wednesday"),
        reschedule_time_slot: str = Query("10:30"),
    ):
        p = require_predictor()
        model = AttendanceModel(
            predictor=p,
            lesson_id=lesson_id,
            lesson_date=lesson_date,
            group=group,
            reschedule_weekday=reschedule_weekday,
            reschedule_time_slot=reschedule_time_slot,
        )
        model.step()
        teacher_agents = [a for a in model.agents if hasattr(a, "policy") and a.policy is not None]
        if not teacher_agents:
            raise HTTPException(status_code=500, detail="Teacher policy not available")
        policy = teacher_agents[0].policy
        return {
            "use_interactive": policy.use_interactive,
            "use_quizzes": policy.use_quizzes,
            "strengthen_attendance_control": policy.strengthen_attendance_control,
            "recommendations": policy.recommendations,
        }

    return app
