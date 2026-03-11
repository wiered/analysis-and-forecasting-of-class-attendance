# Анализ и прогнозирование посещаемости занятий

Десктопное приложение для анализа и прогнозирования посещаемости студентов. Использует модель CatBoost и данные из PostgreSQL для поддержки управленческих решений.

## Возможности

- **Дашборд** — обзор посещаемости
- **Индикаторы риска** — выявление студентов с риском пропусков
- **Прогноз** — вероятность посещения по группам и студентам (с учётом факторов)
- **Сценарный анализ** — оценка эффекта переноса занятий
- **Рекомендации** — подсказки на основе модели
- **API** — панель запуска FastAPI-сервера для доступа к прогнозам по HTTP

## Требования

- Python 3.x
- PostgreSQL с данными посещаемости
- Обученная модель: `attendance_model.cbm` и `feature_names.pkl` (в корне проекта или путь из конфигурации)

## Установка

```bash
pip install -r requirements.txt
```

Создайте файл `.env` в корне проекта:

```env
host=localhost
database=postgres
user=postgres
password=postgres
port=5432
```

При необходимости задайте `LOGGING_LEVEL` и `LOGGING_FORMAT`.

## Запуск

Из корня репозитория (чтобы в PYTHONPATH был `src`):

```bash
python -m src.main
```

или, если `src` в PYTHONPATH:

```bash
python src/main.py
```

При отсутствии подключения к БД или файлов модели приложение откроется с предупреждением; экраны, требующие предиктор, будут недоступны.

## Стек

- **UI:** PySide6 (Qt), pyqtgraph
- **Модель:** CatBoost, scikit-learn, pandas
- **БД:** psycopg2, python-dotenv
- **API:** FastAPI, uvicorn

## Author's comment

if anyone reading this, please help me: i was overrun by an AI model
