"""
Точка входа десктопного приложения (PySide6).
Инициализирует предиктор из db_config; при ошибке показывает сообщение и открывает окно без бэкенда.
"""
import logging
import sys
from logging.config import dictConfig
from typing import Optional

from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory
from PySide6.QtGui import QPalette, QColor

from src.logging_config import LOGGING_CONFIG
from src.settings import db_config
from src.model import AttendancePredictor
from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def setup_fusion_light_theme(app: QApplication) -> None:
    """Оформление в стиле Fusion со светлой цветовой схемой."""
    style = QStyleFactory.create("Fusion")
    if style:
        app.setStyle(style)
    else:
        app.setStyle("Fusion")

    palette = QPalette()
    # Фон и текст
    palette.setColor(QPalette.ColorRole.Window, QColor(243, 243, 243))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(247, 247, 247))
    palette.setColor(QPalette.ColorRole.Text, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(128, 128, 128))
    # Кнопки и акценты
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    # Ссылки и группы
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 102, 204))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(102, 51, 153))
    app.setPalette(palette)


def create_predictor() -> Optional[AttendancePredictor]:
    """Создаёт предиктор; при ошибке возвращает None."""
    try:
        predictor = AttendancePredictor(db_config=db_config)
        logger.info("Предиктор успешно создан")
        return predictor
    except Exception as e:
        logger.exception("Не удалось создать предиктор: %s", e)
        return None


def main() -> None:
    dictConfig(LOGGING_CONFIG)
    logger.info("Запуск UI приложения")
    app = QApplication(sys.argv)
    app.setApplicationName("Посещаемость — поддержка решений")
    setup_fusion_light_theme(app)

    predictor = create_predictor()
    if predictor is None:
        logger.warning("Окно открыто без предиктора")
        QMessageBox.warning(
            None,
            "Нет подключения к бэкенду",
            "Не удалось создать предиктор (нужны БД и файлы attendance_model.cbm, feature_names.pkl).\n"
            "Проверьте .env и наличие файлов модели. Экраны будут недоступны.",
            QMessageBox.StandardButton.Ok,
        )

    window = MainWindow(predictor=predictor)
    window.show()
    logger.info("Главное окно отображено")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
