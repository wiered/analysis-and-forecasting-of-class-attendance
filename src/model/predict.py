import psycopg2
import pandas as pd
from catboost import CatBoostClassifier, Pool
from datetime import datetime, timedelta
import joblib

# ==========================================
# 1. Подключение к PostgreSQL
# ==========================================

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ForKursovik",
        user="postgres",
        password="passwordforDB",
        port="5432"
    )

# ==========================================
# 2. Основная функция формирования признаков
# ==========================================

def build_feature_row(student_id: int, lesson_id: int, lesson_date: str):

    conn = get_connection()
    cursor = conn.cursor()

    lesson_date = datetime.strptime(lesson_date, "%Y-%m-%d")
    four_weeks_ago = lesson_date - timedelta(days=28)

    # Получаем преподавателя занятия
    cursor.execute("""
        SELECT teacher_name
        FROM schedule
        WHERE lesson_id = %s
    """, (lesson_id,))
    teacher_name = cursor.fetchone()[0]

    # Получаем отношение студента к преподавателю
    cursor.execute("""
        SELECT fairness, clarity, sympathy, fear, usefulness
        FROM teacher_relation
        WHERE student_id = %s
        AND teacher_name = %s
    """, (student_id, teacher_name))

    fairness, clarity, sympathy, fear, usefulness = cursor.fetchone()

    # ==========================================
    # 1. Профиль студента
    # ==========================================
    cursor.execute("""
        SELECT gpa, commute_time, works, dorm
        FROM students
        WHERE student_id = %s
    """, (student_id,))
    gpa, commute_time, works, dorm = cursor.fetchone()

    # ==========================================
    # 2. Психология
    # ==========================================
    cursor.execute("""
        SELECT motivation, discipline, anxiety, burnout, extraversion
        FROM psychology
        WHERE student_id = %s
    """, (student_id,))
    motivation, discipline, anxiety, burnout, extraversion = cursor.fetchone()

    # ==========================================
    # 3. Расписание занятия
    # ==========================================
    cursor.execute("""
        SELECT weekday, time_slot, duration
        FROM schedule
        WHERE lesson_id = %s
    """, (lesson_id,))
    weekday, time_slot, duration = cursor.fetchone()

    early_class = 1 if time_slot in ("08:30", "09:00") else 0

    # ==========================================
    # 4. Учебная нагрузка
    # ==========================================
    cursor.execute("""
        SELECT COALESCE(deadlines_count, 0)
        FROM academic_load
        WHERE student_id = %s
        AND week_start <= %s
        AND week_start + interval '7 days' > %s
        LIMIT 1
    """, (student_id, lesson_date, lesson_date))

    row = cursor.fetchone()
    deadlines_count = row[0] if row else 0

    # ==========================================
    # 5. Средняя посещаемость за 4 недели
    # ==========================================
    cursor.execute("""
        SELECT COALESCE(AVG(status), 1.0)
        FROM attendance
        WHERE student_id = %s
        AND date < %s
        AND date >= %s
    """, (student_id, lesson_date, four_weeks_ago))

    avg_attendance_last4 = cursor.fetchone()[0]

    # ==========================================
    # 6. Предыдущий пропуск
    # ==========================================
    cursor.execute("""
        SELECT status
        FROM attendance
        WHERE student_id = %s
        AND date < %s
        ORDER BY date DESC
        LIMIT 1
    """, (student_id, lesson_date))

    row = cursor.fetchone()
    prev_absence = 1 - row[0] if row else 0

    # ==========================================
    # 7. Социальное влияние
    # ==========================================
    cursor.execute("""
        SELECT student_id_2
        FROM social_graph
        WHERE student_id_1 = %s
    """, (student_id,))
    neighbors = [r[0] for r in cursor.fetchall()]

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

    # ==========================================
    # Закрываем соединение
    # ==========================================
    cursor.close()
    conn.close()
    # ==========================================
    # Формируем DataFrame
    # ==========================================
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

model = CatBoostClassifier()
model.load_model("attendance_model.cbm")
feature_names = joblib.load("feature_names.pkl")

def predict_student_attendance(student_id, lesson_id, lesson_date):

    # 1. Формируем признаки
    features = build_feature_row(student_id, lesson_id, lesson_date)

    # 2. Выравниваем порядок колонок
    features = features[feature_names]

    # 3. Категориальные признаки
    categorical_features = ["weekday", "time_slot"]

    # 4. Создаём Pool
    pool = Pool(
        data=features,
        cat_features=categorical_features
    )

    # 5. Вероятность
    proba = model.predict_proba(pool)[0][1]

    # 6. SHAP через Pool
    shap_values = model.get_feature_importance(
        pool,
        type="ShapValues"
    )

    # Убираем базовое значение
    shap_values = shap_values[:, :-1]
    shap_contrib = shap_values[0]

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": shap_contrib
    })

    explanation_df["abs_val"] = explanation_df["contribution"].abs()
    explanation_df = explanation_df.sort_values(
        by="abs_val",
        ascending=False
    )

    top_factors = explanation_df.head(5)

    factors_list = []

    for _, row in top_factors.iterrows():
        direction = (
            "увеличивает риск пропуска"
            if row["contribution"] < 0
            else "снижает риск пропуска"
        )

        factors_list.append({
            "feature": row["feature"],
            "impact": round(row["contribution"], 3),
            "effect": direction
        })

    return {
        "attendance_probability": round(proba, 4),
        "absence_probability": round(1 - proba, 4),
        "top_factors": factors_list
    }

def group_information(group, lesson_id, lesson_date):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
            SELECT student_id, full_name
            FROM students
            WHERE st_group = %s
        """, (group,))
    data = cursor.fetchall()
    for i in data:
        print(i[1])
        result = predict_student_attendance(i[0], lesson_id, lesson_date)
        print("\n=== ПРОГНОЗ ===")
        print("Вероятность посещения:", result["attendance_probability"])
        print("Вероятность пропуска:", result["absence_probability"])

        print("\nОсновные факторы:")
        for f in result["top_factors"]:
            print(f"- {f['feature']} ({f['effect']}, вклад={f['impact']})")

group_information(1, 2, "2026-03-04")






