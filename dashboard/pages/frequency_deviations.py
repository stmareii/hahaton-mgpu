"""Страница: нарушения рекомендации о частоте (не чаще 1 раза в 3 месяца)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_clean, load_raw, compute_intervals, violation_summary, fmt_int, MIN_INTERVAL_DAYS

st.set_page_config(page_title="Нарушения частоты", layout="wide")
st.title("Нарушения рекомендации о частоте")
st.caption(f"Рекомендация: не чаще 1 раза в {MIN_INTERVAL_DAYS} дней (3 месяца)")

clean = load_clean()
raw = load_raw()
intervals = compute_intervals(clean)
summary = violation_summary(intervals)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Нарушений всего", fmt_int(summary["violation_rows"]))
c2.metric("Детей с нарушениями", fmt_int(summary["violation_children"]))
c3.metric("Тестов в один день", fmt_int(summary["same_day_rows"]),
          "явная техошибка", delta_color="inverse")
c4.metric("Медианный интервал", f"{summary['median_interval_violations']:.0f} дн.")

st.divider()

# ---------- Гистограмма интервалов ---------- #
st.subheader("Распределение интервалов между тестами")
df_plot = intervals[intervals["days_since_prev"].notna()].copy()
df_plot["status"] = df_plot["is_violation"].map({True: "Нарушение", False: "Норма"})

fig = px.histogram(
    df_plot, x="days_since_prev", color="status",
    nbins=60, barmode="overlay",
    color_discrete_map={"Нарушение": "#E63946", "Норма": "#2E86AB"},
)
fig.add_vline(x=MIN_INTERVAL_DAYS, line_dash="dash", line_color="black",
              annotation_text=f"{MIN_INTERVAL_DAYS} дней", annotation_position="top")
fig.update_layout(height=380, xaxis_title="Дней между тестами",
                  yaxis_title="Количество", margin=dict(t=20, b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Топы школ / площадок ---------- #
left, right = st.columns(2)

# Топ школ-направивших по нарушениям
# Подтягиваем названия из raw
name_map = (raw[["ogrn_naprav", "name_naprav"]]
            .drop_duplicates("ogrn_naprav")
            .set_index("ogrn_naprav")["name_naprav"])
area_map = (raw[["ogrn_area", "name_area"]]
            .drop_duplicates("ogrn_area")
            .set_index("ogrn_area")["name_area"])

viol = intervals[intervals["is_violation"]]

with left:
    st.subheader("Топ-10 направивших школ по нарушениям")
    top_n = viol["ogrn_naprav"].value_counts().head(10).reset_index()
    top_n.columns = ["ogrn_naprav", "violations"]
    top_n["Название"] = top_n["ogrn_naprav"].map(name_map).fillna("-").str[:80]
    fig = px.bar(top_n, x="violations", y="Название", orientation="h",
                 color_discrete_sequence=["#E63946"])
    fig.update_layout(height=380, margin=dict(t=10, b=0),
                      yaxis=dict(autorange="reversed"),
                      yaxis_title=None, xaxis_title="Нарушений")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Топ-10 площадок по нарушениям")
    top_a = viol["ogrn_area"].value_counts().head(10).reset_index()
    top_a.columns = ["ogrn_area", "violations"]
    top_a["Название"] = top_a["ogrn_area"].map(area_map).fillna("-").str[:80]
    fig = px.bar(top_a, x="violations", y="Название", orientation="h",
                 color_discrete_sequence=["#E63946"])
    fig.update_layout(height=380, margin=dict(t=10, b=0),
                      yaxis=dict(autorange="reversed"),
                      yaxis_title=None, xaxis_title="Нарушений")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Детальная таблица ---------- #
st.subheader("Все нарушения - детали")
show_only_same_day = st.checkbox("Показать только тесты в один день (0 дней)", value=False)

table = viol.copy()
if show_only_same_day:
    table = table[table["days_since_prev"] == 0]

table_view = table[[
    "last_name", "first_name", "middle_name", "bdate",
    "prev_date", "test_date", "days_since_prev",
    "ogrn_naprav", "ogrn_area", "class", "variant", "result",
]].rename(columns={
    "prev_date": "Предыдущий тест",
    "test_date": "Текущий тест",
    "days_since_prev": "Интервал (дн.)",
})

st.dataframe(table_view, use_container_width=True, height=400)
st.download_button(
    "Скачать нарушения (CSV)",
    table_view.to_csv(index=False).encode("utf-8-sig"),
    "violations.csv",
    "text/csv",
)
