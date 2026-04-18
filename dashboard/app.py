"""Точка входа дашборда. Запуск: streamlit run app.py"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="Контроль качества тестирования",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/overview.py",              title="Обзор"),
    st.Page("pages/frequency_deviations.py",  title="Нарушения частоты"),
    st.Page("pages/anomalies.py",             title="Аномалии"),
    st.Page("pages/comparison.py",            title="Было / Стало"),
    st.Page("pages/details.py",               title="Детали и выгрузка"),
])
pg.run()
