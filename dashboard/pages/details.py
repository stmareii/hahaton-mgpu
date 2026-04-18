"""Страница: фильтруемая таблица всех записей + выгрузка."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd

from utils import load_clean, load_raw, compute_intervals, fmt_int

st.title("Детали и выгрузка")

source = st.radio(
    "Источник данных",
    ["Проанализированный (hakaton_analyzed.csv)", "Исходный (hakaton.csv)"],
    horizontal=True,
)
df = load_clean() if source.startswith("Проанализированный") else load_raw()

# Фильтры
with st.expander("Фильтры", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        date_range = st.date_input(
            "Период тестирования",
            value=(df["test_date"].min().date(), df["test_date"].max().date()),
        )
    with c2:
        classes = st.multiselect("Классы", sorted(df["class"].dropna().unique()))
    with c3:
        results = st.multiselect("Результат", df["result"].dropna().unique())

    c4, c5 = st.columns(2)
    with c4:
        search = st.text_input("Поиск по фамилии ребёнка")
    with c5:
        only_violations = st.checkbox("Только нарушения частоты", value=False)

    # Флаги доступны только в очищенном датасете
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    active_flags = []
    if flag_cols:
        st.write("Фильтр по флагам аномалий:")
        flag_row = st.columns(len(flag_cols))
        flag_labels = {
            "flag_suspicious_id": "Подозрительный id",
            "flag_frequency_violation": "Нарушение частоты",
            "flag_age_anomaly": "Аномальный возраст",
            "flag_parent_too_young": "Опекун слишком молод",
            "flag_parent_child_id_match": "id совпадает с опекуном",
        }
        for col_widget, flag in zip(flag_row, flag_cols):
            label = flag_labels.get(flag, flag)
            if col_widget.checkbox(label, value=False, key=flag):
                active_flags.append(flag)

filtered = df.copy()
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["test_date"] >= start) & (filtered["test_date"] <= end)]
if classes:
    filtered = filtered[filtered["class"].isin(classes)]
if results:
    filtered = filtered[filtered["result"].isin(results)]
if search:
    filtered = filtered[filtered["last_name"].fillna("").str.contains(search.upper(), case=False)]
if only_violations:
    if "flag_frequency_violation" in filtered.columns:
        filtered = filtered[filtered["flag_frequency_violation"]]
    else:
        intervals = compute_intervals(df)
        violating_keys = set(intervals[intervals["is_violation"]]["child_key"])
        filtered = filtered[filtered["child_key"].isin(violating_keys)]
for flag in active_flags:
    filtered = filtered[filtered[flag]]

st.caption(f"Отобрано: **{fmt_int(len(filtered))}** из {fmt_int(len(df))}")
st.dataframe(filtered, use_container_width=True, height=500)

st.download_button(
    "Скачать отфильтрованное (CSV)",
    filtered.to_csv(index=False).encode("utf-8-sig"),
    "filtered.csv",
    "text/csv",
)
