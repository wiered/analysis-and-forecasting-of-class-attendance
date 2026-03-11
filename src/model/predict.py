import psycopg2
import pandas as pd
from catboost import CatBoostClassifier, Pool
from datetime import datetime, timedelta
import joblib
from dataclasses import dataclass
from typing import List, Optional

__all__ = ["AttendancePredictor", "StudentPrediction", "GroupSummary", "GroupFactorSummary", "FactorImpact"]

# ==========================================================
# ======================= DATACLASSES ======================
# ==========================================================

@dataclass
class FactorImpact:
    """
    Описывает вклад одного признака в прогноз модели.

    Attributes:
        feature (str): Название признака.
        impact (float): SHAP-вклад признака в модель.
        effect (str): Интерпретация влияния:
                      - "увеличивает риск пропуска"
                      - "снижает риск пропуска"
    """
    feature: str
    impact: float
    effect: str


@dataclass
class StudentPrediction:
    """
    Результат прогноза посещаемости для одного студента.

    Attributes:
        student_id (int): ID студента.
        attendance_probability (float): Вероятность посещения занятия.
        absence_probability (float): Вероятность пропуска занятия.
        top_factors (List[FactorImpact]): Топ-5 факторов влияния.
        full_name (Optional[str]): ФИО студента (если доступно).
    """
    student_id: int
    attendance_probability: float
    absence_probability: float
    top_factors: List[FactorImpact]
    full_name: Optional[str] = None


@dataclass
class GroupFactorSummary:
    """
    Сводное влияние признака на уровне группы.

    Attributes:
        feature (str): Название признака.
        students_affected (int): Количество студентов,
                                 у которых этот фактор в топ-5.
        avg_impact (float): Средний абсолютный вклад признака.
    """
    feature: str
    students_affected: int
    avg_impact: float


@dataclass
class GroupSummary:
    """
    Итоговая статистика прогноза по группе.

    Attributes:
        group (int): Номер группы.
        lesson_id (int): ID занятия.
        lesson_date (str): Дата занятия.
        total_students (int): Количество студентов.
        avg_attendance_probability (float): Средняя вероятность посещения.
        students_at_risk (int): Количество студентов с риском > 50%.
        risk_percentage (float): Процент студентов в зоне риска.
        top_group_factors (List[GroupFactorSummary]):
            Самые частые факторы риска по группе.
    """
    group: int
    lesson_id: int
    lesson_date: str
    total_students: int
    avg_attendance_probability: float
    students_at_risk: int
    risk_percentage: float
    top_group_factors: List[GroupFactorSummary]


# ==========================================================
# ================== ATTENDANCE PREDICTOR ==================
# ==========================================================

class AttendancePredictor:
    """
    Класс для прогнозирования посещаемости студентов
    с использованием модели CatBoost.
    """

    def __init__(
        self,
        db_config: Optional[dict] = None,
        model_path: str = "attendance_model.cbm",
        features_path: str = "feature_names.pkl"
    ):
        """
        Инициализация предиктора.

        Args:
            db_config (dict, optional): Параметры подключения к БД.
                Пример:
                    {
                        "host": "localhost",
                        "user": "postgres",
                        "password": "postgres",
                        "port": "5432",
                        "database": "postgres"
                    }
            model_path (str): Путь к файлу обученной модели.
            features_path (str): Путь к сохранённому списку признаков.
        """

        if not db_config:
            db_config = {
                "host": "localhost",
                "database": "postgres",
                "user": "postgres",
                "password": "postgres",
                "port": "5432"
            }

        self.db_config = db_config

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        self.feature_names: List[str] = joblib.load(features_path)

        self.categorical_features = ["weekday", "time_slot"]

    # ======================================================
    # ================== DB CONNECTION =====================
    # ======================================================
    def _get_connection(self):
        """Создаёт соединение с PostgreSQL."""
        return psycopg2.connect(**self.db_config)

    # ======================================================
    # ================= FEATURE BUILDING ===================
    # ======================================================

    def _build_feature_row(
            self,
            student_id: int,
            lesson_id: int,
            lesson_date: str
    ) -> pd.DataFrame:
        """
        Формирует строку признаков для одного студента.

        Args:
            student_id (int): ID студента.
            lesson_id (int): ID занятия.
            lesson_date (str): Дата занятия (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame с одной строкой признаков.
        """

        conn = self._get_connection()
        cursor = conn.cursor()

        lesson_date_dt = datetime.strptime(lesson_date, "%Y-%m-%d")
        four_weeks_ago = lesson_date_dt - timedelta(days=28)

        # Преподаватель
        cursor.execute("""
            SELECT teacher_name
            FROM schedule
            WHERE lesson_id = %s
        """, (lesson_id,))
        row = cursor.fetchone()
        teacher_name = row[0] if row else None

        if not teacher_name:
            cursor.close()
            conn.close()
            raise ValueError(f"Преподаватель не найден для занятия {lesson_id}")

        # Отношение к преподавателю
        cursor.execute("""
            SELECT fairness, clarity, sympathy, fear, usefulness
            FROM teacher_relation
            WHERE student_id = %s
            AND teacher_name = %s
        """, (student_id, teacher_name))
        fairness, clarity, sympathy, fear, usefulness = cursor.fetchone()

        # Профиль студента
        cursor.execute("""
            SELECT gpa, commute_time, works, dorm
            FROM students
            WHERE student_id = %s
        """, (student_id,))
        gpa, commute_time, works, dorm = cursor.fetchone()

        # Психология
        cursor.execute("""
            SELECT motivation, discipline, anxiety, burnout, extraversion
            FROM psychology
            WHERE student_id = %s
        """, (student_id,))
        motivation, discipline, anxiety, burnout, extraversion = cursor.fetchone()

        # Расписание
        cursor.execute("""
            SELECT weekday, time_slot, duration
            FROM schedule
            WHERE lesson_id = %s
        """, (lesson_id,))
        weekday, time_slot, duration = cursor.fetchone()

        early_class = 1 if time_slot in ("08:30", "09:00") else 0

        # Учебная нагрузка
        cursor.execute("""
            SELECT COALESCE(deadlines_count, 0)
            FROM academic_load
            WHERE student_id = %s
            AND week_start <= %s
            AND week_start + interval '7 days' > %s
            LIMIT 1
        """, (student_id, lesson_date_dt, lesson_date_dt))
        row = cursor.fetchone()
        deadlines_count = row[0] if row else 0

        # Средняя посещаемость
        cursor.execute("""
            SELECT COALESCE(AVG(status), 1.0)
            FROM attendance
            WHERE student_id = %s
            AND date < %s
            AND date >= %s
        """, (student_id, lesson_date_dt, four_weeks_ago))
        row = cursor.fetchone()
        avg_attendance_last4 = row[0] if row else 1.0

        # Предыдущий пропуск
        cursor.execute("""
            SELECT status
            FROM attendance
            WHERE student_id = %s
            AND date < %s
            ORDER BY date DESC
            LIMIT 1
        """, (student_id, lesson_date_dt))
        row = cursor.fetchone()
        prev_absence = 1 - row[0] if row else 0

        # Социальное влияние
        cursor.execute("""
            SELECT student_id_2
            FROM social_graph
            WHERE student_id_1 = %s
        """, (student_id,))
        rows = cursor.fetchall()
        neighbors = [r[0] for r in rows] if rows else []

        if neighbors:
            cursor.execute("""
                SELECT COALESCE(AVG(status), 1.0)
                FROM attendance
                WHERE student_id = ANY(%s)
                AND lesson_id = %s
            """, (neighbors, lesson_id))
            friends_mean_attendance = cursor.fetchone()[0]
        else:
            friends_mean_attendance = 1.0
        cursor.close()
        conn.close()

        feature_dict = {
            "gpa": gpa,
            "commute_time": commute_time,
            "works": works,
            "dorm": dorm,
            "motivation": motivation,
            "discipline": discipline,
            "anxiety": anxiety,
            "burnout": burnout,
            "extraversion": extraversion,
            "deadlines_count": deadlines_count,
            "avg_attendance_last4": avg_attendance_last4,
            "prev_absence": prev_absence,
            "weekday": weekday,
            "time_slot": time_slot,
            "duration": duration,
            "early_class": early_class,
            "friends_mean_attendance": friends_mean_attendance,
            "fairness": fairness,
            "clarity": clarity,
            "sympathy": sympathy,
            "fear": fear,
            "usefulness": usefulness
        }

        return pd.DataFrame([feature_dict])

    # ======================================================
    # ================= STUDENT PREDICTION =================
    # ======================================================

    def predict_student(
        self,
        student_id: int,
        lesson_id: int,
        lesson_date: str
    ) -> StudentPrediction:
        """
        Прогнозирует посещаемость для одного студента.

        Returns:
            StudentPrediction: Структурированный результат прогноза.
        """

        features = self._build_feature_row(student_id, lesson_id, lesson_date)
        features = features[self.feature_names]

        pool = Pool(
            data=features,
            cat_features=self.categorical_features
        )

        proba = self.model.predict_proba(pool)[0][1]

        shap_values = self.model.get_feature_importance(
            pool,
            type="ShapValues"
        )[:, :-1][0]

        explanation_df = pd.DataFrame({
            "feature": self.feature_names,
            "contribution": shap_values
        })

        explanation_df["abs_val"] = explanation_df["contribution"].abs()
        explanation_df = explanation_df.sort_values(
            by="abs_val",
            ascending=False
        )

        top_factors = explanation_df.head(5)

        factors_list: List[FactorImpact] = []

        for _, row in top_factors.iterrows():
            direction = (
                "увеличивает риск пропуска"
                if row["contribution"] < 0
                else "снижает риск пропуска"
            )

            factors_list.append(
                FactorImpact(
                    feature=row["feature"],
                    impact=round(row["contribution"], 3),
                    effect=direction
                )
            )

        return StudentPrediction(
            student_id=student_id,
            attendance_probability=round(proba, 4),
            absence_probability=round(1 - proba, 4),
            top_factors=factors_list
        )

    # ======================================================
    # ================= GROUP PREDICTION ===================
    # ======================================================

    def predict_group(
        self,
        group: int,
        lesson_id: int,
        lesson_date: str,
        verbose: bool = True
    ) -> List[StudentPrediction]:
        """
        Прогнозирует посещаемость для всей группы.

        Returns:
            List[StudentPrediction]: Список прогнозов по студентам.
        """

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT student_id, full_name
            FROM students
            WHERE st_group = %s
        """, (group,))
        students = cursor.fetchall()

        cursor.close()
        conn.close()

        results: List[StudentPrediction] = []

        for student_id, full_name in students:
            result = self.predict_student(student_id, lesson_id, lesson_date)
            result.full_name = full_name
            if verbose:
                print(f"\n{full_name} (ID: {student_id})")
                print("=" * 50)
                print("Вероятность посещения:", result.attendance_probability)
                print("Вероятность пропуска:", result.absence_probability)

            results.append(result)

        return results

    # ======================================================
    # ================= GROUP SUMMARY ======================
    # ======================================================

    def get_group_summary(
        self,
        group: int,
        lesson_id: int,
        lesson_date: str
    ) -> GroupSummary:
        """
        Возвращает агрегированную статистику по группе.
        """

        results = self.predict_group(group, lesson_id, lesson_date, verbose=False)

        total_students = len(results)
        avg_attendance = sum(r.attendance_probability for r in results) / total_students
        students_at_risk = sum(1 for r in results if r.absence_probability > 0.5)

        all_factors = []
        for result in results:
            all_factors.extend(result.top_factors)

        factor_counts = {}

        for factor in all_factors:
            if factor.feature not in factor_counts:
                factor_counts[factor.feature] = {"count": 0, "total_impact": 0}

            factor_counts[factor.feature]["count"] += 1
            factor_counts[factor.feature]["total_impact"] += abs(factor.impact)

        top_factors = sorted(
            factor_counts.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:5]

        group_factor_objects = [
            GroupFactorSummary(
                feature=feature,
                students_affected=data["count"],
                avg_impact=round(data["total_impact"] / data["count"], 3)
            )
            for feature, data in top_factors
        ]

        return GroupSummary(
            group=group,
            lesson_id=lesson_id,
            lesson_date=lesson_date,
            total_students=total_students,
            avg_attendance_probability=round(avg_attendance, 4),
            students_at_risk=students_at_risk,
            risk_percentage=round(students_at_risk / total_students * 100, 2),
            top_group_factors=group_factor_objects
        )


# ==========================================================
# ======================== EXAMPLE =========================
# ==========================================================

if __name__ == "__main__":
    predictor = AttendancePredictor()

    student_result = predictor.predict_student(
        student_id=1,
        lesson_id=2,
        lesson_date="2026-03-04"
    )
    print("\nПрогноз для студента:")
    print(student_result)

    group_summary = predictor.get_group_summary(
        group=1,
        lesson_id=2,
        lesson_date="2026-03-04"
    )
    print("\nСводка по группе:")
    print(group_summary)