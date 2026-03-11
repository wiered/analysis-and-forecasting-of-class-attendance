"""Панель состояния API для взаимодействия агентов (вкл/выкл, порт, справка по эндпоинтам)."""
import logging
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QFormLayout,
    QGroupBox,
    QTextEdit,
)
from PySide6.QtCore import QObject, Signal, QThread

from src.model import AttendancePredictor

logger = logging.getLogger(__name__)


class ApiServerRunner(QObject):
    """Запуск uvicorn в потоке."""
    started = Signal(str)
    stopped = Signal()
    error = Signal(str)

    def __init__(self, predictor: Optional[AttendancePredictor], host: str, port: int):
        super().__init__()
        self.predictor = predictor
        self.host = host
        self.port = port
        self._server = None

    def run(self) -> None:
        try:
            from src.api.server import create_app
            import uvicorn
            app = create_app(self.predictor)
            config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
            self._server = uvicorn.Server(config)
            self._server.install_signal_handlers = lambda: None
            self.started.emit(f"http://{self.host}:{self.port}")
            logger.info("API сервер запущен на %s:%s", self.host, self.port)
            self._server.run()
        except Exception as e:
            logger.exception("Ошибка запуска API сервера: %s", e)
            self.error.emit(str(e))
        finally:
            self.stopped.emit()

    def shutdown(self) -> None:
        if self._server:
            logger.info("Остановка API сервера")
            self._server.should_exit = True


class ApiPanelWidget(QWidget):
    def __init__(self, predictor: Optional[AttendancePredictor] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.predictor = predictor
        self._runner = None
        self._thread = None

        layout = QVBoxLayout(self)

        status_group = QGroupBox("Состояние API")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("API не запущен.")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        form = QFormLayout()
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8765)
        form.addRow("Порт:", self.port_spin)
        status_layout.addLayout(form)

        self.start_btn = QPushButton("Запустить API")
        self.stop_btn = QPushButton("Остановить API")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        status_layout.addWidget(self.start_btn)
        status_layout.addWidget(self.stop_btn)
        layout.addWidget(status_group)

        help_group = QGroupBox("Эндпоинты для агентов")
        help_layout = QVBoxLayout(help_group)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "GET /group_summary?group=1&lesson_id=2&lesson_date=2026-03-04\n"
            "GET /predict_student?student_id=1&lesson_id=2&lesson_date=2026-03-04\n"
            "GET /predict_group?group=1&lesson_id=2&lesson_date=2026-03-04\n"
            "GET /reschedule_effect?group=1&lesson_id=2&lesson_date=...&new_weekday=Wednesday&new_time_slot=10:30\n"
            "GET /recommendations?group=1&lesson_id=2&lesson_date=2026-03-04\n\n"
            "При отсутствии предиктора API возвращает 503."
        )
        help_layout.addWidget(help_text)
        layout.addWidget(help_group)

    def _on_start(self) -> None:
        if not self.predictor:
            self.status_label.setText("Предиктор недоступен. Запуск API невозможен.")
            return
        port = self.port_spin.value()
        self._runner = ApiServerRunner(self.predictor, "127.0.0.1", port)
        self._thread = QThread()
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.started.connect(self._on_server_started)
        self._runner.stopped.connect(self._on_server_stopped)
        self._runner.error.connect(self._on_server_error)
        self._runner.stopped.connect(self._thread.quit)
        self._thread.start()
        self.start_btn.setEnabled(False)
        self.port_spin.setEnabled(False)

    def _on_server_started(self, url: str) -> None:
        self.status_label.setText(f"API запущен: {url}")
        self.stop_btn.setEnabled(True)

    def _on_server_stopped(self) -> None:
        self.status_label.setText("API остановлен.")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.port_spin.setEnabled(True)
        self._thread = None
        self._runner = None

    def _on_server_error(self, msg: str) -> None:
        self.status_label.setText(f"Ошибка: {msg}")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.port_spin.setEnabled(True)

    def _on_stop(self) -> None:
        if self._runner:
            self._runner.shutdown()
