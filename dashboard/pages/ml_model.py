"""Страница: результаты Isolation Forest."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_ml_results, fmt_int

st.title("ML-модель: Isolation Forest")

df = load_ml_results()

if df is None:
    st.warning(
        "Файл `data/dataset_with_model_scores.csv` не найден. "
        "Запустите ноутбук `model_isolation_forest_1.ipynb` целиком, чтобы получить результаты модели."
    )
    st.stop()

# ── KPI ──────────────────────────────────────────────────────────────────────
total = len(df)
model_anom = int(df["model_anomaly"].sum())
rule_anom  = int(df["has_anomaly"].sum())
both       = int((df["model_anomaly"] & df["has_anomaly"]).sum())
model_only = int((df["model_anomaly"] & ~df["has_anomaly"]).sum())
combined   = int(df["combined_anomaly"].sum())
recall_pct = both / rule_anom * 100 if rule_anom else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Аномалий по модели",         fmt_int(model_anom))
c2.metric("Новые находки модели",        fmt_int(model_only), help="Модель нашла, правила пропустили")
c3.metric("Recall по правилам",          f"{recall_pct:.1f}%", help="Какую долю аномалий правил подтвердила модель")
c4.metric("Чистых (объединённый флаг)", fmt_int(total - combined))

st.divider()

# ── Распределение anomaly_score ───────────────────────────────────────────────
st.subheader("Распределение anomaly score")

df_plot = df.copy()
df_plot["Тип"] = df_plot["model_anomaly"].map({True: "Аномалия", False: "Норма"})

fig = px.histogram(
    df_plot,
    x="anomaly_score",
    color="Тип",
    nbins=80,
    color_discrete_map={"Аномалия": "#E63946", "Норма": "#2E86AB"},
    labels={"anomaly_score": "Anomaly score", "count": "Записей"},
    barmode="overlay",
    opacity=0.75,
)
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Чем ниже anomaly score — тем сильнее запись отличается от «нормальной». "
    "Порог отсечения соответствует contamination=0.05 при обучении."
)

st.divider()

# ── Ключевая находка: same_school ─────────────────────────────────────────────
st.subheader("Ключевая находка: same_school")

new_finds = df[df["model_anomaly"] & ~df["has_anomaly"]]
pct_new = new_finds["same_school"].mean() * 100 if len(new_finds) else 0
pct_all = df["same_school"].mean() * 100

col1, col2 = st.columns(2)
col1.metric("same_school в новых находках", f"{pct_new:.1f}%")
col2.metric("same_school в целом по датасету", f"{pct_all:.1f}%")

st.info(
    "**same_school** — случай, когда школа, направившая ребёнка, "
    "сама же является площадкой тестирования. "
    "Потенциальный конфликт интересов: организация контролирует результат сама себя."
)

st.divider()

# ── Матрица пересечения ───────────────────────────────────────────────────────
st.subheader("Пересечение: модель vs правила")

rule_only  = int((~df["model_anomaly"] & df["has_anomaly"]).sum())
neither    = int((~df["model_anomaly"] & ~df["has_anomaly"]).sum())

matrix = pd.DataFrame(
    {
        "Модель: ДА":  [fmt_int(both),    fmt_int(model_only)],
        "Модель: НЕТ": [fmt_int(rule_only), fmt_int(neither)],
    },
    index=["Правила: ДА", "Правила: НЕТ"],
)
st.table(matrix)

st.divider()

# ── Топ аномальных записей ────────────────────────────────────────────────────
st.subheader("Топ-20 самых аномальных записей")

display_cols = [c for c in [
    "last_name", "first_name", "class",
    "child_age", "n_tests", "n_classes", "min_gap",
    "same_school", "has_anomaly", "model_anomaly", "anomaly_score",
] if c in df.columns]

top20 = df.nsmallest(20, "anomaly_score")[display_cols].copy()

rename = {
    "last_name": "Фамилия", "first_name": "Имя", "class": "Класс",
    "child_age": "Возраст", "n_tests": "Кол-во тестов", "n_classes": "Классов",
    "min_gap": "Мин. интервал (дн.)", "same_school": "Та же школа",
    "has_anomaly": "Аномалия (правила)", "model_anomaly": "Аномалия (модель)",
    "anomaly_score": "Score",
}
top20 = top20.rename(columns={k: v for k, v in rename.items() if k in top20.columns})
if "Возраст" in top20.columns:
    top20["Возраст"] = top20["Возраст"].round(1)
if "Score" in top20.columns:
    top20["Score"] = top20["Score"].round(3)

st.dataframe(top20, use_container_width=True, hide_index=True)

st.download_button(
    label="Скачать топ-20 CSV",
    data=top20.to_csv(index=False).encode("utf-8"),
    file_name="top20_anomalies_ml.csv",
    mime="text/csv",
)
