"""Главное окно приложения с вкладками."""
import logging
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QMessageBox,
)
from PySide6.QtCore import Qt

from src.model import AttendancePredictor

from .dashboard import DashboardWidget
from .risk_indicators import RiskIndicatorsWidget
from .forecast import ForecastWidget
from .scenario import ScenarioWidget
from .recommendations import RecommendationsWidget
from .simulation import SimulationWidget
from .api_panel import ApiPanelWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, predictor: Optional[AttendancePredictor] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        logger.debug("Создание главного окна, predictor=%s", "есть" if predictor else "нет")
        self.setWindowTitle("Посещаемость — поддержка управленческих решений")
        self.setMinimumSize(700, 500)
        self.resize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.addTab(DashboardWidget(predictor), "Дашборд")
        self.tabs.addTab(RiskIndicatorsWidget(predictor), "Индикаторы риска")
        self.tabs.addTab(ForecastWidget(predictor), "Прогноз")
        self.tabs.addTab(ScenarioWidget(predictor), "Сценарный анализ")
        self.tabs.addTab(RecommendationsWidget(predictor), "Рекомендации")
        self.tabs.addTab(SimulationWidget(predictor), "Симуляция")
        self.tabs.addTab(ApiPanelWidget(predictor), "API")
        layout.addWidget(self.tabs)
