"""Страница: фильтруемая таблица всех записей + выгрузка."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd

from utils import load_clean, load_raw, compute_intervals, fmt_int

st.set_page_config(page_title="Детали", page_icon="📋", layout="wide")
st.title("Детали и выгрузка")

source = st.radio(
    "Источник данных",
    ["Очищенный (without_duplicates.csv)", "Исходный (hakaton.csv)"],
    horizontal=True,
)
df = load_clean() if source.startswith("Очищенный") else load_raw()

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
        only_violations = st.checkbox("Только нарушения частоты", value=False)
    with c5:
        search = st.text_input("Поиск по фамилии ребёнка")

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
    intervals = compute_intervals(df)
    violating_keys = set(intervals[intervals["is_violation"]]["child_key"])
    filtered = filtered[filtered["child_key"].isin(violating_keys)]

st.caption(f"Отобрано: **{fmt_int(len(filtered))}** из {fmt_int(len(df))}")
st.dataframe(filtered, use_container_width=True, height=500)

st.download_button(
    "Скачать отфильтрованное (CSV)",
    filtered.to_csv(index=False).encode("utf-8-sig"),
    "filtered.csv",
    "text/csv",
)
