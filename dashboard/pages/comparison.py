"""Страница: сравнение было/стало (hakaton.csv ---> hakaton_cleaned.csv из cleaning.ipynb)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_raw, DATA_DIR, fmt_int

st.title("Было / Стало")
st.caption("Сравнение исходного датасета (hakaton.csv) с результатом очистки (hakaton_cleaned.csv)")

FINAL_PATH = DATA_DIR / "hakaton_cleaned.csv"

@st.cache_data(show_spinner="Загружаю hakaton_cleaned.csv…")
def load_final() -> pd.DataFrame:
    df = pd.read_csv(FINAL_PATH, low_memory=False)
    for col in ["test_date", "bdate", "guard_bdate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df["variant"] = pd.to_numeric(df["variant"], errors="coerce")
    return df

if not FINAL_PATH.exists():
    st.warning(
        "Файл `data/hakaton_cleaned.csv` не найден. "
        "Запусти все ячейки `cleaning.ipynb` чтобы его сгенерировать."
    )
    st.stop()

raw = load_raw()
final = load_final()

# ── KPI ───────────────────────────────────────────────────────────────────
st.subheader("Основные показатели")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Записей исходно", fmt_int(len(raw)))
c2.metric(
    "Записей после очистки",
    fmt_int(len(final)),
    f"−{fmt_int(len(raw) - len(final))}",
    delta_color="inverse",
)
c3.metric("Уникальных детей (исходно)", fmt_int(raw["child_key"].nunique()))
c4.metric(
    "Уникальных детей (после)",
    fmt_int(final["person_id"].nunique()) if "person_id" in final.columns else "-",
)

st.divider()

# ── Что убрали ─────────────────────────────────────────────────────────────
st.subheader("Что изменилось при очистке")

raw_n = len(raw)
final_n = len(final)

# Считаем реальные числа из данных
raw_result_norm = raw.copy()
raw_result_norm["result"] = raw_result_norm["result"].str.strip().str.upper()
dedup_cols = [c for c in raw_result_norm.columns if c not in ["our_number", "name_naprav", "name_area", "child_key"]]
after_full_dedup = raw_result_norm.drop_duplicates(subset=dedup_cols).shape[0]
full_dups = raw_n - after_full_dedup

ogrn_bad = int((raw["ogrn_naprav"].astype(str).str.len() != 13).sum())

# Остальные шаги берём из разницы (variant invalid, key dups, child_too_old, МИШИН)
other = raw_n - final_n - full_dups - ogrn_bad

changes = pd.DataFrame([
    {"Операция": "Полные дубликаты",                    "Удалено строк": full_dups},
    {"Операция": "Некорректный ОГРН",                   "Удалено строк": ogrn_bad},
    {"Операция": "Невалидный variant, child_too_old и др.", "Удалено строк": other},
])
changes["Удалено строк"] = changes["Удалено строк"].apply(lambda x: f"−{x}")
st.dataframe(changes, use_container_width=True, hide_index=True)

st.markdown("""
**Без удалений строк (нормализация и исправления):**
- `id_doc`: убраны префиксы №, N, − ---> записи одного ребёнка объединяются корректно
- `variant`: нестандартные форматы приведены к числовому коду
- `class = "2-5"` ---> `"3"`
- Исправлены даты рождения 4 детей с некорректным годом (2025 ---> реальный)
- Исправлены 3 записи где `bdate == guard_bdate`
- Пустые `id_doc` заполнены по совпадению ФИО+ДР или новым сгенерированным id
""")

st.divider()

# ── Распределение результатов ──────────────────────────────────────────────
st.subheader("Результаты тестирований")

left, right = st.columns(2)
with left:
    st.caption("Исходный датасет")
    r_raw = raw["result"].str.strip().str.upper().value_counts().reset_index()
    r_raw.columns = ["result", "count"]
    fig = px.pie(r_raw, values="count", names="result", hole=0.5,
                 color_discrete_sequence=["#2E86AB", "#E63946"])
    fig.update_layout(height=300, margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.caption("После очистки")
    r_fin = final["result"].value_counts().reset_index()
    r_fin.columns = ["result", "count"]
    fig = px.pie(r_fin, values="count", names="result", hole=0.5,
                 color_discrete_sequence=["#2E86AB", "#E63946"])
    fig.update_layout(height=300, margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Распределение по классам ───────────────────────────────────────────────
st.subheader("Распределение по классам")

left, right = st.columns(2)

def class_chart(df, title):
    cls = df["class"].value_counts().reset_index()
    cls.columns = ["class", "count"]
    cls["order"] = pd.to_numeric(cls["class"], errors="coerce")
    cls = cls.sort_values("order")
    fig = px.bar(cls, x="class", y="count", color_discrete_sequence=["#2E86AB"])
    fig.update_layout(height=320, margin=dict(t=20, b=0), xaxis_title=None, title=title)
    return fig

with left:
    st.plotly_chart(class_chart(raw, "Исходный"), use_container_width=True)
with right:
    st.plotly_chart(class_chart(final, "После очистки"), use_container_width=True)

st.divider()

# ── Аномалии после очистки ─────────────────────────────────────────────────
anomaly_cols = [c for c in ["freq_violation", "same_day_test", "multi_class_jump",
                             "id_equals_guard", "age_class_mismatch", "guard_too_young"]
                if c in final.columns]

if anomaly_cols:
    st.subheader("Аномалии в очищенном датасете")
    anom_df = pd.DataFrame({
        "Аномалия": anomaly_cols,
        "Записей": [int(final[c].sum()) for c in anomaly_cols],
    })
    anom_df["% от датасета"] = (anom_df["Записей"] / len(final) * 100).round(2)
    anom_df = anom_df.sort_values("Записей", ascending=False).reset_index(drop=True)

    # Итоговая строка - уникальные записи хотя бы с одной аномалией
    total_with_anomaly = int(final[anomaly_cols].any(axis=1).sum())
    total_pct = round(total_with_anomaly / len(final) * 100, 2)
    totals = pd.DataFrame([{
        "Аномалия": "Итого (уникальных записей с ≥1 аномалией)",
        "Записей": total_with_anomaly,
        "% от датасета": total_pct,
    }])
    anom_df = pd.concat([anom_df, totals], ignore_index=True)

    st.dataframe(anom_df, use_container_width=True, hide_index=True)
