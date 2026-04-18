"""Страница: нарушения рекомендации о частоте (не чаще 1 раза в 3 месяца)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px  # используется для гистограммы

from utils import load_clean, load_raw, compute_intervals, violation_summary, fmt_int, MIN_INTERVAL_DAYS

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
c4.metric("Медианный интервал", f"{summary["median_interval_violations"]:.0f} дн.")

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
def _most_common_name(series):
    return series.value_counts().index[0]

name_map = raw.groupby("ogrn_naprav")["name_naprav"].agg(_most_common_name)
area_map = raw.groupby("ogrn_area")["name_area"].agg(_most_common_name)

viol = intervals[intervals["is_violation"]]

st.subheader("Топ-10 направивших школ по нарушениям")
top_n = viol["ogrn_naprav"].value_counts().head(10).reset_index()
top_n.columns = ["ogrn_naprav", "violations"]
top_n["ОГРН"] = top_n["ogrn_naprav"].astype(str)
top_n["Название"] = top_n["ogrn_naprav"].apply(lambda x: name_map.get(str(x), "—"))
top_n["Нарушений"] = top_n["violations"]
st.dataframe(top_n[["Название", "ОГРН", "Нарушений"]], use_container_width=True, hide_index=True)

st.subheader("Топ-10 площадок по нарушениям")
top_a = viol["ogrn_area"].value_counts().head(10).reset_index()
top_a.columns = ["ogrn_area", "violations"]
top_a["ОГРН"] = top_a["ogrn_area"].astype(str)
top_a["Название"] = top_a["ogrn_area"].apply(lambda x: area_map.get(str(x), "—"))
top_a["Нарушений"] = top_a["violations"]
st.dataframe(top_a[["Название", "ОГРН", "Нарушений"]], use_container_width=True, hide_index=True)

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
