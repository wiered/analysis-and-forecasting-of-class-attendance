"""
Конфигурация приложения из .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Корень репозитория (родитель каталога src/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Логирование
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
LOGGING_FORMAT = os.getenv("LOGGING_FORMAT", "standard")

# Параметры БД
DB_HOST = os.getenv("host", "localhost")
DB_NAME = os.getenv("database", "postgres")
DB_USER = os.getenv("user", "postgres")
DB_PASSWORD = os.getenv("password", "postgres")
DB_PORT = os.getenv("port", "5432")

# Словарь для передачи в psycopg2 / AttendancePredictor
db_config = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT,
}
