import logging
import os
import psycopg2
import pandas as pd
from catboost import CatBoostClassifier, Pool
from datetime import datetime, timedelta
import joblib
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .constants import WEEKDAYS, TIME_SLOTS

logger = logging.getLogger(__name__)

# Корень репозитория (родитель каталога src/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

__all__ = [
    "AttendancePredictor",
    "StudentPrediction",
    "GroupSummary",
    "GroupFactorSummary",
    "FactorImpact",
    "RescheduleEffect",
    "BestSlotResult",
    "WEEKDAYS",
    "TIME_SLOTS",
]

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


@dataclass
class RescheduleEffect:
    """
    Ожидаемый эффект от переноса занятия (для агента деканата).

    Attributes:
        avg_attendance_before (float): Средняя вероятность посещения до переноса.
        avg_attendance_after (float): Средняя вероятность посещения после переноса.
        delta (float): Разница (after - before).
        risk_pct_before (float): Процент студентов в зоне риска до переноса.
        risk_pct_after (float): Процент студентов в зоне риска после переноса.
    """
    avg_attendance_before: float
    avg_attendance_after: float
    delta: float
    risk_pct_before: float
    risk_pct_after: float


@dataclass
class BestSlotResult:
    """
    Результат поиска лучшего слота для переноса занятия.

    Attributes:
        best_weekday (str): День недели лучшего слота.
        best_time_slot (str): Время лучшего слота.
        reschedule_effect (RescheduleEffect): Эффект переноса в этот слот.
        current_weekday (Optional[str]): Текущий день занятия (из БД), или None.
        current_time_slot (Optional[str]): Текущее время занятия (из БД), или None.
    """
    best_weekday: str
    best_time_slot: str
    reschedule_effect: RescheduleEffect
    current_weekday: Optional[str] = None
    current_time_slot: Optional[str] = None


# Ранние и поздние слоты для эвристик подбора (индексы в TIME_SLOTS)
_EARLY_TIME_SLOTS = ("08:30", "09:00")
_LATE_TIME_SLOT = "16:00"
_MID_WEEKDAYS = ("Wednesday", "Thursday")


def _build_smart_candidates(
    current_weekday: Optional[str],
    current_time_slot: Optional[str],
    group_summary: Optional[GroupSummary] = None,
    max_candidates: int = 12,
) -> List[Tuple[str, str]]:
    """
    Строит короткий список перспективных слотов для переноса по эвристикам:
    ранняя пара -> более позднее время; поздняя -> более раннее; средняя -> соседние.
    Учитывает top_group_factors (early_class, time_slot, weekday) для набора дней/времён.
    """
    factor_names = set()
    if group_summary and group_summary.top_group_factors:
        factor_names = {gf.feature for gf in group_summary.top_group_factors}

    current_pair = (current_weekday, current_time_slot) if (current_weekday and current_time_slot) else None

    def exclude_current(wd: str, ts: str) -> bool:
        return current_pair is None or (wd, ts) != current_pair

    # Выбор времён по типу текущего слота
    if current_time_slot in _EARLY_TIME_SLOTS:
        # Ранняя пара -> пробуем более позднее время
        preferred_times = [t for t in TIME_SLOTS if t not in _EARLY_TIME_SLOTS]
    elif current_time_slot == _LATE_TIME_SLOT:
        # Поздняя пара -> более раннее время (не 08:30/09:00 и не 16:00)
        preferred_times = ["10:30", "12:00", "14:00"]
    elif current_time_slot in ("10:30", "12:00", "14:00"):
        # Средняя пара -> соседние по времени
        idx = TIME_SLOTS.index(current_time_slot)
        neighbor_idxs = [i for i in (idx - 1, idx + 1) if 0 <= i < len(TIME_SLOTS)]
        preferred_times = [TIME_SLOTS[i] for i in neighbor_idxs]
        if not preferred_times:
            preferred_times = ["10:30", "12:00", "14:00"]
    else:
        preferred_times = ["10:30", "12:00", "14:00"]

    # Дни: тот же, затем 1–2 «хороших» (среда, четверг)
    use_extra_weekdays = (current_weekday is not None and current_weekday in WEEKDAYS) or "weekday" in factor_names
    if current_weekday and current_weekday in WEEKDAYS:
        weekdays_same_first = [current_weekday]
        for w in _MID_WEEKDAYS:
            if w != current_weekday and w not in weekdays_same_first:
                weekdays_same_first.append(w)
                if len(weekdays_same_first) >= 3:
                    break
        if not use_extra_weekdays:
            weekdays_same_first = [current_weekday]
    else:
        weekdays_same_first = list(_MID_WEEKDAYS)[:3]

    # Сначала «тот же день, другое время», затем «другой день»
    candidates: List[Tuple[str, str]] = []
    for wd in weekdays_same_first:
        for ts in preferred_times:
            if exclude_current(wd, ts):
                candidates.append((wd, ts))
            if len(candidates) >= max_candidates:
                return candidates

    # Если мало набрали — добавить слоты с другими днями/временами
    if len(candidates) < max_candidates and use_extra_weekdays:
        for wd in WEEKDAYS:
            if wd in weekdays_same_first:
                continue
            for ts in preferred_times[:3]:
                if exclude_current(wd, ts) and (wd, ts) not in candidates:
                    candidates.append((wd, ts))
                    if len(candidates) >= max_candidates:
                        return candidates

    return candidates[:max_candidates]


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
        db_config: dict,
        model_path: str = "attendance_model.cbm",
        features_path: str = "feature_names.pkl",
    ):
        """
        Инициализация предиктора.

        Args:
            db_config (dict): Параметры подключения к БД. Пример:
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
        logger.debug("AttendancePredictor.__init__: начало, db_config keys=%s", list(db_config.keys()))
        self.db_config = db_config

        # Ищем файлы модели: сначала в CWD, затем в корне репозитория
        def _resolve(path: str) -> str:
            if os.path.isabs(path) and os.path.isfile(path):
                return path
            if os.path.isfile(path):
                return path
            root_path = os.path.join(_REPO_ROOT, path)
            if os.path.isfile(root_path):
                return root_path
            return path  # оставляем как есть — CatBoost/joblib выдадут ошибку с путём

        model_path = _resolve(model_path)
        features_path = _resolve(features_path)
        logger.debug("AttendancePredictor.__init__: model_path=%s, features_path=%s", model_path, features_path)

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        logger.debug("AttendancePredictor.__init__: модель загружена")

        self.feature_names: List[str] = joblib.load(features_path)
        logger.debug("AttendancePredictor.__init__: feature_names загружены, count=%s", len(self.feature_names))

        self.categorical_features = ["weekday", "time_slot"]
        logger.debug("AttendancePredictor.__init__: готов")

    # ======================================================
    # ================== DB CONNECTION =====================
    # ======================================================
    def _get_connection(self):
        """Создаёт соединение с PostgreSQL."""
        logger.debug("_get_connection: открытие соединения с БД")
        conn = psycopg2.connect(**self.db_config)
        logger.debug("_get_connection: соединение установлено")
        return conn

    # ======================================================
    # ================= FEATURE BUILDING ===================
    # ======================================================

    def _build_feature_row(
            self,
            student_id: int,
            lesson_id: int,
            lesson_date: str,
            schedule_override: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Формирует строку признаков для одного студента.

        Args:
            student_id (int): ID студента.
            lesson_id (int): ID занятия.
            lesson_date (str): Дата занятия (YYYY-MM-DD).
            schedule_override (dict, optional): Подстановка расписания с ключами
                weekday, time_slot (значения как в БД). Не меняет данные в БД.

        Returns:
            pd.DataFrame: DataFrame с одной строкой признаков.
        """
        logger.debug("_build_feature_row: начало student_id=%s lesson_id=%s lesson_date=%s", student_id, lesson_id, lesson_date)
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
            logger.warning("_build_feature_row: преподаватель не найден lesson_id=%s", lesson_id)
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

        if schedule_override:
            weekday = schedule_override.get("weekday", weekday)
            time_slot = schedule_override.get("time_slot", time_slot)

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
        logger.debug("_build_feature_row: признаки собраны student_id=%s", student_id)

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
        lesson_date: str,
        schedule_override: Optional[dict] = None
    ) -> StudentPrediction:
        """
        Прогнозирует посещаемость для одного студента.

        Returns:
            StudentPrediction: Структурированный результат прогноза.
        """
        logger.debug("predict_student: начало student_id=%s lesson_id=%s lesson_date=%s", student_id, lesson_id, lesson_date)
        features = self._build_feature_row(
            student_id, lesson_id, lesson_date, schedule_override=schedule_override
        )
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

        result = StudentPrediction(
            student_id=student_id,
            attendance_probability=round(proba, 4),
            absence_probability=round(1 - proba, 4),
            top_factors=factors_list
        )
        logger.debug("predict_student: готов student_id=%s proba=%.4f", student_id, proba)
        return result

    # ======================================================
    # ================= GROUP PREDICTION ===================
    # ======================================================

    def predict_group(
        self,
        group: int,
        lesson_id: int,
        lesson_date: str,
        verbose: bool = True,
        schedule_override: Optional[dict] = None
    ) -> List[StudentPrediction]:
        """
        Прогнозирует посещаемость для всей группы.

        Returns:
            List[StudentPrediction]: Список прогнозов по студентам.
        """
        logger.debug("predict_group: начало group=%s lesson_id=%s lesson_date=%s", group, lesson_id, lesson_date)
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
        logger.debug("predict_group: загружен список студентов, count=%s", len(students))

        results: List[StudentPrediction] = []

        for idx, (student_id, full_name) in enumerate(students):
            result = self.predict_student(
                student_id, lesson_id, lesson_date,
                schedule_override=schedule_override
            )
            result.full_name = full_name
            logger.debug("predict_group: студент %s/%s id=%s прогноз=%.4f", idx + 1, len(students), student_id, result.attendance_probability)
            if verbose:
                print(f"\n{full_name} (ID: {student_id})")
                print("=" * 50)
                print("Вероятность посещения:", result.attendance_probability)
                print("Вероятность пропуска:", result.absence_probability)

            results.append(result)

        logger.debug("predict_group: готово group=%s, результатов=%s", group, len(results))
        return results

    # ======================================================
    # ================= GROUP SUMMARY ======================
    # ======================================================

    def get_group_summary(
        self,
        group: int,
        lesson_id: int,
        lesson_date: str,
        schedule_override: Optional[dict] = None
    ) -> GroupSummary:
        """
        Возвращает агрегированную статистику по группе.
        """
        logger.info(
            "get_group_summary: начало group=%s lesson_id=%s lesson_date=%s schedule_override=%s",
            group, lesson_id, lesson_date, schedule_override,
        )

        results = self.predict_group(
            group, lesson_id, lesson_date,
            verbose=False,
            schedule_override=schedule_override
        )
        logger.debug("get_group_summary: predict_group вернул %s результатов", len(results))

        total_students = len(results)
        if total_students == 0:
            logger.warning("get_group_summary: группа пуста group=%s", group)
        avg_attendance = sum(r.attendance_probability for r in results) / total_students if total_students else 0.0
        students_at_risk = sum(1 for r in results if r.absence_probability > 0.5)
        logger.debug(
            "get_group_summary: агрегация total_students=%s avg_attendance=%.4f students_at_risk=%s",
            total_students, avg_attendance, students_at_risk,
        )

        all_factors = []
        for result in results:
            all_factors.extend(result.top_factors)
        logger.debug("get_group_summary: собрано факторов (всего записей) %s", len(all_factors))

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
        logger.debug("get_group_summary: топ факторов по группе: %s", [f[0] for f in top_factors])

        group_factor_objects = [
            GroupFactorSummary(
                feature=feature,
                students_affected=data["count"],
                avg_impact=round(data["total_impact"] / data["count"], 3)
            )
            for feature, data in top_factors
        ]

        summary = GroupSummary(
            group=group,
            lesson_id=lesson_id,
            lesson_date=lesson_date,
            total_students=total_students,
            avg_attendance_probability=round(avg_attendance, 4),
            students_at_risk=students_at_risk,
            risk_percentage=round(students_at_risk / total_students * 100, 2) if total_students else 0.0,
            top_group_factors=group_factor_objects
        )
        logger.info(
            "get_group_summary: готово group=%s total_students=%s avg=%.4f risk=%s",
            group, total_students, summary.avg_attendance_probability, summary.students_at_risk,
        )
        return summary

    @staticmethod
    def group_summary_from_predictions(
        results: List[StudentPrediction],
        group: int,
        lesson_id: int,
        lesson_date: str,
    ) -> GroupSummary:
        """
        Строит GroupSummary по готовому списку прогнозов (без повторного вызова predict_group).
        """
        total_students = len(results)
        if total_students == 0:
            return GroupSummary(
                group=group,
                lesson_id=lesson_id,
                lesson_date=lesson_date,
                total_students=0,
                avg_attendance_probability=0.0,
                students_at_risk=0,
                risk_percentage=0.0,
                top_group_factors=[],
            )
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
            reverse=True,
        )[:5]
        group_factor_objects = [
            GroupFactorSummary(
                feature=feature,
                students_affected=data["count"],
                avg_impact=round(data["total_impact"] / data["count"], 3),
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
            top_group_factors=group_factor_objects,
        )

    # ======================================================
    # ============== LESSON SCHEDULE =======================
    # ======================================================

    def get_lesson_schedule(self, lesson_id: int) -> Optional[Tuple[str, str, int]]:
        """
        Возвращает текущее расписание занятия из БД.

        Args:
            lesson_id: ID занятия.

        Returns:
            (weekday, time_slot, duration) или None, если запись не найдена.
            duration в минутах (int).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT weekday, time_slot, duration
                FROM schedule
                WHERE lesson_id = %s
            """, (lesson_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            weekday, time_slot, duration = row
            duration_int = int(duration) if duration is not None else 90
            return (weekday, time_slot, duration_int)
        finally:
            cursor.close()
            conn.close()

    # ======================================================
    # ============== RESCHEDULE EFFECT =====================
    # ======================================================

    def get_reschedule_effect(
        self,
        group: int,
        lesson_id: int,
        lesson_date: str,
        new_weekday: str,
        new_time_slot: str
    ) -> RescheduleEffect:
        """
        Ожидаемый эффект от переноса занятия (для агента деканата).

        Сравнивает среднюю вероятность посещения и долю риска
        при текущем расписании и при подстановке new_weekday, new_time_slot.

        Args:
            group: Номер группы.
            lesson_id: ID занятия.
            lesson_date: Дата занятия (YYYY-MM-DD).
            new_weekday: День недели после переноса (как в БД).
            new_time_slot: Время после переноса (как в БД, напр. "10:30").

        Returns:
            RescheduleEffect: avg_attendance_before/after, delta, risk_pct_before/after.
        """
        logger.debug("get_reschedule_effect: начало group=%s new_weekday=%s new_time_slot=%s", group, new_weekday, new_time_slot)
        summary_before = self.get_group_summary(
            group, lesson_id, lesson_date, schedule_override=None
        )
        logger.debug("get_reschedule_effect: summary_before получена")
        summary_after = self.get_group_summary(
            group, lesson_id, lesson_date,
            schedule_override={"weekday": new_weekday, "time_slot": new_time_slot}
        )
        logger.debug("get_reschedule_effect: summary_after получена")
        delta = summary_after.avg_attendance_probability - summary_before.avg_attendance_probability
        logger.debug("get_reschedule_effect: delta=%.4f", delta)
        return RescheduleEffect(
            avg_attendance_before=summary_before.avg_attendance_probability,
            avg_attendance_after=summary_after.avg_attendance_probability,
            delta=round(delta, 4),
            risk_pct_before=summary_before.risk_percentage,
            risk_pct_after=summary_after.risk_percentage,
        )

    def find_best_reschedule_slot(
        self,
        group: int,
        lesson_id: int,
        lesson_date: str,
        candidate_slots: Optional[List[Tuple[str, str]]] = None,
        smart_search: bool = True,
    ) -> BestSlotResult:
        """
        Находит лучший слот для переноса занятия по критерию максимума delta
        (рост посещаемости); при равенстве — минимум risk_pct_after.

        Текущий слот занятия (из БД) исключается из кандидатов.
        При smart_search=True и candidate_slots=None используется сокращённый
        список кандидатов по эвристикам (ранняя/поздняя пара, факторы группы).
        Расчёт «до» выполняется один раз (1+N вызовов get_group_summary).

        Args:
            group: Номер группы.
            lesson_id: ID занятия.
            lesson_date: Дата занятия (YYYY-MM-DD).
            candidate_slots: Список (weekday, time_slot) или None — тогда
                при smart_search=True строится умный список, иначе WEEKDAYS × TIME_SLOTS.
            smart_search: При True и candidate_slots=None строить короткий список по эвристикам.

        Returns:
            BestSlotResult: лучший слот, эффект и текущий слот (если есть).
        """
        current = self.get_lesson_schedule(lesson_id)
        current_pair: Optional[Tuple[str, str]] = None
        current_weekday: Optional[str] = None
        current_time_slot: Optional[str] = None
        if current is not None:
            current_weekday, current_time_slot, _ = current
            current_pair = (current_weekday, current_time_slot)

        # Один раз сводка «до»
        summary_before = self.get_group_summary(
            group, lesson_id, lesson_date, schedule_override=None
        )

        if candidate_slots is None:
            if smart_search:
                candidates = _build_smart_candidates(
                    current_weekday,
                    current_time_slot,
                    group_summary=summary_before,
                    max_candidates=12,
                )
            else:
                candidates = [
                    (wd, ts) for wd in WEEKDAYS for ts in TIME_SLOTS
                    if current_pair is None or (wd, ts) != current_pair
                ]
        else:
            candidates = [
                (wd, ts) for (wd, ts) in candidate_slots
                if current_pair is None or (wd, ts) != current_pair
            ]

        if not candidates:
            eff = RescheduleEffect(
                avg_attendance_before=summary_before.avg_attendance_probability,
                avg_attendance_after=summary_before.avg_attendance_probability,
                delta=0.0,
                risk_pct_before=summary_before.risk_percentage,
                risk_pct_after=summary_before.risk_percentage,
            )
            return BestSlotResult(
                best_weekday=current_weekday or "Wednesday",
                best_time_slot=current_time_slot or "10:30",
                reschedule_effect=eff,
                current_weekday=current_weekday,
                current_time_slot=current_time_slot,
            )

        best_weekday, best_time_slot = candidates[0]
        summary_after_0 = self.get_group_summary(
            group, lesson_id, lesson_date,
            schedule_override={"weekday": best_weekday, "time_slot": best_time_slot},
        )
        delta_0 = summary_after_0.avg_attendance_probability - summary_before.avg_attendance_probability
        best_effect = RescheduleEffect(
            avg_attendance_before=summary_before.avg_attendance_probability,
            avg_attendance_after=summary_after_0.avg_attendance_probability,
            delta=round(delta_0, 4),
            risk_pct_before=summary_before.risk_percentage,
            risk_pct_after=summary_after_0.risk_percentage,
        )
        for wd, ts in candidates[1:]:
            summary_after = self.get_group_summary(
                group, lesson_id, lesson_date,
                schedule_override={"weekday": wd, "time_slot": ts},
            )
            delta = summary_after.avg_attendance_probability - summary_before.avg_attendance_probability
            eff = RescheduleEffect(
                avg_attendance_before=summary_before.avg_attendance_probability,
                avg_attendance_after=summary_after.avg_attendance_probability,
                delta=round(delta, 4),
                risk_pct_before=summary_before.risk_percentage,
                risk_pct_after=summary_after.risk_percentage,
            )
            if eff.delta > best_effect.delta or (
                eff.delta == best_effect.delta
                and eff.risk_pct_after < best_effect.risk_pct_after
            ):
                best_weekday, best_time_slot = wd, ts
                best_effect = eff

        return BestSlotResult(
            best_weekday=best_weekday,
            best_time_slot=best_time_slot,
            reschedule_effect=best_effect,
            current_weekday=current_weekday,
            current_time_slot=current_time_slot,
        )


# ==========================================================
# ======================== EXAMPLE =========================
# ==========================================================

if __name__ == "__main__":
    predictor = AttendancePredictor(
        db_config={
            "host": "localhost",
            "database": "postgres",
            "user": "postgres",
            "password": "postgres",
            "port": "5432",
        }
    )

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