import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import joblib

# ===============================
# 1. Загрузка данных
# ===============================
data = pd.read_csv("src/model/dataset.csv")

print("Размер датасета:", data.shape)

# ===============================
# 2. Приведение BOOLEAN к 0/1
# ===============================
bool_columns = ["works", "dorm"]

for col in bool_columns:
    if col in data.columns:
        data[col] = data[col].map({
            True: 1, False: 0,
            "t": 1, "f": 0,
            "True": 1, "False": 0
        })

# ===============================
# 3. Удаляем ID-признаки
# ===============================
drop_columns = ["id", "student_id", "lesson_id", "lesson_date"]

for col in drop_columns:
    if col in data.columns:
        data = data.drop(columns=[col])

# ===============================
# 4. Разделяем X и y
# ===============================
X = data.drop(columns=["target"])
y = data["target"]

# ===============================
# 5. Категориальные признаки
# ===============================
categorical_features = ["weekday", "time_slot"]

# ===============================
# 6. Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# 7. Инициализация модели
# ===============================
model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100,
    random_seed=42
)

# ===============================
# 8. Обучение
# ===============================
model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_test, y_test)
)

# ===============================
# 9. Предсказания
# ===============================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ===============================
# 10. Метрики
# ===============================
print("\n=== Метрики модели ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ===============================
# 11. Топ-10 важных признаков
# ===============================
feature_importance = model.get_feature_importance()

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_importance
}).sort_values(by="importance", ascending=False)

print("\n=== Топ-10 важных признаков ===")
print(importance_df.head(10))

# ===============================
# 12. Сохраняем модель
# ===============================
model.save_model("attendance_model.cbm")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\nМодель успешно обучена и сохранена.")