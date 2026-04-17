"""Страница: все типы аномалий с выгрузкой."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_clean, load_raw, detect_anomalies, fmt_int

st.set_page_config(page_title="Аномалии", layout="wide")
st.title("Аномалии в данных")
st.caption("Технические ошибки, логические противоречия и некорректные записи")

clean = load_clean()
raw = load_raw()
anomalies = detect_anomalies(clean, raw)

# Свод по всем аномалиям
st.subheader("Сводка")
summary_df = pd.DataFrame({
    "Тип аномалии": list(anomalies.keys()),
    "Записей": [len(v) for v in anomalies.values()],
})
summary_df["% от всего"] = (summary_df["Записей"] / len(clean) * 100).round(2)
summary_df = summary_df.sort_values("Записей", ascending=False).reset_index(drop=True)

left, right = st.columns([2, 3])
with left:
    st.dataframe(summary_df, use_container_width=True, hide_index=True, height=420)
with right:
    fig = px.bar(
        summary_df, x="Записей", y="Тип аномалии", orientation="h",
        color="Записей", color_continuous_scale="Reds",
    )
    fig.update_layout(
        height=420, margin=dict(t=10, b=0),
        yaxis=dict(autorange="reversed"), yaxis_title=None,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Детализация по каждой аномалии
st.subheader("Детализация")
anomaly_descriptions = {
    "id_doc_отрицательный": "Идентификатор начинается с `-`. Признак ошибки генерации/импорта.",
    "id_doc_слишком_короткий": "Длина `id_doc` < 6 символов - нереалистично для реального документа.",
    "дубли_тестов_в_один_день": "Один ребёнок с одинаковыми атрибутами, но двумя разными `our_number` в один день.",
    "ОГРН_направ_несколько_названий": "Справочник школ грязный: один ОГРН указан с разными названиями.",
    "ОГРН_площадка_несколько_названий": "Аналогично для площадок тестирования.",
    "один_id_doc_разные_ФИО": "`id_doc` не уникален - одному значению соответствуют разные ФИО+bdate.",
    "больше_2_опекунов": "Ребёнок с >2 разными опекунами (дядя/тётя/бабушка?).",
    "некорректный_класс": "Класс не в диапазоне 1–11 (например, `2-5`).",
    "возраст_не_соответствует_классу": "Возраст ребёнка на дату теста не соответствует указанному классу.",
    "опекун_моложе_ребёнка": "Дата рождения опекуна >= даты рождения ребёнка - грубая ошибка.",
    "опекун_слишком_молод": "Разница возраста опекун-ребёнок < 14 лет - нереалистично для родителя.",
}

selected = st.selectbox(
    "Выбери тип аномалии",
    options=list(anomalies.keys()),
    format_func=lambda k: f"{k}  -  {len(anomalies[k])} записей",
)

st.write(f"**Описание:** {anomaly_descriptions.get(selected, '-')}")

df = anomalies[selected]
st.write(f"Найдено записей: **{fmt_int(len(df))}**")
st.dataframe(df, use_container_width=True, height=400)

if len(df) > 0:
    st.download_button(
        f"Скачать `{selected}` (CSV)",
        df.to_csv(index=False).encode("utf-8-sig"),
        f"{selected}.csv",
        "text/csv",
    )
