"""Страница: все типы аномалий с выгрузкой."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_clean, load_raw, detect_anomalies, fmt_int

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
    "Один ребёнок — разные классы": "Один ребёнок сдавал тесты, будучи записан в разные классы.",
    "Нарушение частоты (< 90 дней)": "Ребёнок написал тест повторно менее чем через 90 дней после предыдущего.",
    "Тест написан несколько раз в один день": "Один ребёнок сдавал тест более одного раза в один и тот же день.",
    "Возраст не соответствует классу": "Возраст ребёнка на дату теста не соответствует указанному классу (отклонение > 3 лет).",
    "id_doc ребёнка совпадает с id родителя": "`id_doc` ребёнка совпадает с `guard_id_doc` родителя — вероятно, скопирован по ошибке.",
    "Пустой id_doc": "`id_doc` отсутствует или равен `-`.",
    "Пустой guard_id_doc": "`guard_id_doc` отсутствует у записи.",
    "Некорректный вариант теста": "Значение `variant` записано в формате даты или не является числом.",
    "Родитель слишком молод (разница < 18 лет)": "Разница возрастов родителя и ребёнка меньше 18 лет — нереалистично.",
    "Ребёнок слишком юный (< 5 лет)": "Возраст ребёнка на момент теста меньше 5 лет.",
    "Ребёнок слишком старый (> 20 лет)": "Возраст ребёнка на момент теста больше 20 лет.",
    "Дата рождения ребёнка совпадает с датой родителя": "Дата рождения ребёнка и родителя совпадают — грубая ошибка ввода.",
    "Некорректный класс": "Класс записан не как целое число (например, `2-5`).",
    "Тест написан до рождения ребёнка": "Дата теста раньше даты рождения ребёнка — логическое противоречие.",
    "Некорректный ОГРН направившей организации": "ОГРН направившей организации не соответствует стандарту (не 13 цифр).",
    "ОГРН направ. несколько названий": "Один ОГРН направившей организации фигурирует с разными названиями.",
    "ОГРН площадка несколько названий": "Один ОГРН площадки тестирования фигурирует с разными названиями.",
}

selected = st.selectbox(
    "Выбери тип аномалии",
    options=list(anomalies.keys()),
    format_func=lambda k: f"{k}  -  {len(anomalies[k])} записей",
)

st.write(f"**Описание:** {anomaly_descriptions.get(selected, "-")}")

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
