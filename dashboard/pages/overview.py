"""Страница: общий обзор датасета."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_clean, load_raw, compute_intervals, fmt_int

st.title("Анализ соблюдения рекомендаций тестирования")

clean = load_clean()
raw = load_raw()
intervals = compute_intervals(clean)

# ---------- KPI ---------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Всего тестов", fmt_int(len(clean)))
c2.metric("Уникальных детей", fmt_int(clean["child_key"].nunique()))
c3.metric("Школ-направивших", fmt_int(clean["ogrn_naprav"].nunique()))
c4.metric("Площадок", fmt_int(clean["ogrn_area"].nunique()))

c5, c6, c7, c8 = st.columns(4)
c5.metric(
    "Период",
    f"{clean['test_date'].min():%d.%m.%Y}",
    f"до {clean['test_date'].max():%d.%m.%Y}",
    delta_color="off",
)
c6.metric(
    "Нарушений частоты",
    fmt_int(int(intervals["is_violation"].sum())),
    f"{intervals['is_violation'].sum() / len(intervals) * 100:.2f}% от всех тестов",
    delta_color="inverse",
)
c7.metric(
    "Детей с нарушениями",
    fmt_int(intervals[intervals["is_violation"]]["child_key"].nunique()),
)
c8.metric(
    "Дубли из исходника",
    fmt_int(len(raw) - len(clean)),
    "удалил сокомандник",
    delta_color="off",
)

st.divider()

# ---------- Распределения ---------- #
left, right = st.columns(2)

with left:
    st.subheader("Результаты тестирований")
    fig = px.pie(
        clean["result"].value_counts().reset_index(),
        values="count",
        names="result",
        hole=0.5,
        color_discrete_sequence=["#2E86AB", "#E63946"],
    )
    fig.update_layout(height=350, margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Распределение по классам")
    cls = clean["class"].value_counts().reset_index()
    cls.columns = ["class", "count"]
    cls["order"] = pd.to_numeric(cls["class"], errors="coerce")
    cls = cls.sort_values("order")
    fig = px.bar(cls, x="class", y="count", color_discrete_sequence=["#2E86AB"])
    fig.update_layout(height=350, margin=dict(t=20, b=0, l=0, r=0), xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Таймлайн ---------- #
st.subheader("Тестирования по времени")
timeline = (
    clean.groupby(clean["test_date"].dt.to_period("W").dt.to_timestamp())
    .size()
    .reset_index(name="count")
)
fig = px.area(
    timeline, x="test_date", y="count",
    color_discrete_sequence=["#2E86AB"],
)
fig.update_layout(height=280, margin=dict(t=10, b=0, l=0, r=0),
                  xaxis_title=None, yaxis_title="Тестов в неделю")
st.plotly_chart(fig, use_container_width=True)

st.info(
    "В левом меню: **Нарушения частоты**, **Аномалии**, **Было / Стало**, **Детали / Выгрузка**."
)
