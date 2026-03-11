"""Графики на pyqtgraph для виджетов анализа посещаемости."""
from typing import List

import pyqtgraph as pg

from src.model import (
    GroupFactorSummary,
    RescheduleEffect,
    StudentPrediction,
    FactorImpact,
)


def _clear_plot(plot_widget: pg.PlotItem) -> None:
    plot_widget.clear()


def plot_reschedule_attendance(
    plot_widget: pg.PlotItem,
    effect: RescheduleEffect,
) -> None:
    """График «До / После»: средняя вероятность посещения."""
    _clear_plot(plot_widget)
    plot_widget.setTitle("Средняя вероятность посещения")
    plot_widget.setLabel("left", "Вероятность")
    plot_widget.setLabel("bottom", "")
    heights = [effect.avg_attendance_before, effect.avg_attendance_after]
    bar = pg.BarGraphItem(
        x=[1, 2],
        height=heights,
        width=0.5,
        brush=pg.mkColor(70, 130, 180),
    )
    plot_widget.addItem(bar)
    plot_widget.getAxis("bottom").setTicks([[(1, "До"), (2, "После")]])
    plot_widget.getViewBox().setXRange(0.4, 2.6)
    plot_widget.getViewBox().setYRange(0, 1.05)


def plot_reschedule_risk(
    plot_widget: pg.PlotItem,
    effect: RescheduleEffect,
) -> None:
    """График «До / После»: доля студентов в зоне риска, %."""
    _clear_plot(plot_widget)
    plot_widget.setTitle("Доля в зоне риска, %")
    plot_widget.setLabel("left", "%")
    plot_widget.setLabel("bottom", "")
    heights = [effect.risk_pct_before, effect.risk_pct_after]
    bar = pg.BarGraphItem(
        x=[1, 2],
        height=heights,
        width=0.5,
        brush=pg.mkColor(220, 20, 60),
    )
    plot_widget.addItem(bar)
    plot_widget.getAxis("bottom").setTicks([[(1, "До"), (2, "После")]])
    plot_widget.getViewBox().setXRange(0.4, 2.6)
    plot_widget.getViewBox().setYRange(0, max(heights) * 1.15 if heights else 1)


def plot_probability_histogram(
    plot_widget: pg.PlotItem,
    probabilities: List[float],
    title: str = "Распределение вероятностей посещения",
) -> None:
    """Гистограмма распределения вероятностей посещения по группе."""
    _clear_plot(plot_widget)
    plot_widget.setTitle(title)
    plot_widget.setLabel("left", "Число студентов")
    plot_widget.setLabel("bottom", "Вероятность посещения")

    if not probabilities:
        return
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    counts = [0] * (len(bins) - 1)
    for p in probabilities:
        for i in range(len(bins) - 1):
            if bins[i] <= p < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if p >= 1.0:
                counts[-1] += 1

    x = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    width = 0.15
    bar = pg.BarGraphItem(x=x, height=counts, width=width, brush=pg.mkColor(100, 149, 237))
    plot_widget.addItem(bar)
    plot_widget.getViewBox().setXRange(-0.1, 1.1)
    plot_widget.getViewBox().setYRange(0, max(counts) * 1.15 if counts else 1)


def plot_top_factors(
    plot_widget: pg.PlotItem,
    factors: List[GroupFactorSummary],
    title: str = "Топ факторов по группе",
) -> None:
    """Столбчатая диаграмма: фактор → число затронутых студентов."""
    _clear_plot(plot_widget)
    plot_widget.setTitle(title)
    plot_widget.setLabel("left", "Студентов затронуто")
    plot_widget.setLabel("bottom", "")

    if not factors:
        return
    n = len(factors)
    x_pos = list(range(n))
    heights = [f.students_affected for f in factors]
    labels = [f.feature[:16] + ("…" if len(f.feature) > 16 else "") for f in factors]
    bar = pg.BarGraphItem(x=x_pos, height=heights, width=0.5, brush=pg.mkColor(72, 61, 139))
    plot_widget.addItem(bar)
    plot_widget.getAxis("bottom").setTicks([[(i, labels[i]) for i in range(n)]])
    plot_widget.getViewBox().setXRange(-0.5, n)
    plot_widget.getViewBox().setYRange(0, max(heights) * 1.15 if heights else 1)


def plot_risk_split(
    plot_widget: pg.PlotItem,
    at_risk: int,
    not_at_risk: int,
    title: str = "Студенты в зоне риска",
) -> None:
    """Два столбца: в зоне риска / вне зоны риска."""
    _clear_plot(plot_widget)
    plot_widget.setTitle(title)
    plot_widget.setLabel("left", "Число студентов")
    plot_widget.setLabel("bottom", "")

    x = [1, 2]
    heights = [at_risk, not_at_risk]
    colors = [pg.mkColor(220, 20, 60), pg.mkColor(34, 139, 34)]
    for i, (xi, h) in enumerate(zip(x, heights)):
        bar = pg.BarGraphItem(x=[xi], height=[h], width=0.5, brush=colors[i])
        plot_widget.addItem(bar)
    plot_widget.getAxis("bottom").setTicks([[(1, "В зоне риска"), (2, "Вне зоны риска")]])
    plot_widget.getViewBox().setXRange(0.4, 2.6)
    plot_widget.getViewBox().setYRange(0, max(heights) * 1.15 if heights else 1)


def plot_student_ranking(
    plot_widget: pg.PlotItem,
    results: List[StudentPrediction],
    top_n: int = 20,
    title: str = "Рейтинг по вероятности посещения",
) -> None:
    """Столбчатая диаграмма: студенты по убыванию вероятности посещения (топ N)."""
    _clear_plot(plot_widget)
    plot_widget.setTitle(title)
    plot_widget.setLabel("left", "Вероятность посещения")
    plot_widget.setLabel("bottom", "")

    if not results:
        return
    sorted_results = sorted(results, key=lambda r: r.attendance_probability, reverse=True)[:top_n]
    n = len(sorted_results)
    x_pos = list(range(n))
    probs = [r.attendance_probability for r in sorted_results]
    labels = [
        (r.full_name or f"ID {r.student_id}")[:14] + ("…" if len((r.full_name or str(r.student_id))) > 14 else "")
        for r in sorted_results
    ]
    bar = pg.BarGraphItem(x=x_pos, height=probs, width=0.5, brush=pg.mkColor(70, 130, 180))
    plot_widget.addItem(bar)
    plot_widget.getAxis("bottom").setTicks([[(i, labels[i]) for i in range(n)]])
    plot_widget.getViewBox().setXRange(-0.5, n)
    plot_widget.getViewBox().setYRange(0, 1.05)


def plot_student_factors(
    plot_widget: pg.PlotItem,
    factors: List[FactorImpact],
    title: str = "Вклад факторов в прогноз",
) -> None:
    """Столбцы: фактор → |вклад|; цвет по влиянию (увеличивает риск — красный, снижает — зелёный)."""
    _clear_plot(plot_widget)
    plot_widget.setTitle(title)
    plot_widget.setLabel("left", "|Вклад|")
    plot_widget.setLabel("bottom", "")

    if not factors:
        return
    n = len(factors)
    x_pos = list(range(n))
    abs_impacts = [abs(f.impact) for f in factors]
    labels = [f.feature[:16] + ("…" if len(f.feature) > 16 else "") for f in factors]
    colors = [
        pg.mkColor(220, 20, 60) if "увеличивает" in f.effect else pg.mkColor(34, 139, 34)
        for f in factors
    ]
    for i in range(n):
        bar = pg.BarGraphItem(
            x=[x_pos[i]],
            height=[abs_impacts[i]],
            width=0.4,
            brush=colors[i],
        )
        plot_widget.addItem(bar)
    plot_widget.getAxis("bottom").setTicks([[(i, labels[i]) for i in range(n)]])
    max_impact = max(abs_impacts) or 0.01
    plot_widget.getViewBox().setXRange(-0.5, n)
    plot_widget.getViewBox().setYRange(0, max_impact * 1.2)
