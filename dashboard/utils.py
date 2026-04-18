"""
Общие функции для дашборда: загрузка данных, нормализация, детекторы аномалий.
Всё закэшировано через st.cache_data, так что страницы работают быстро.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

def _find_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here / "data", here.parent / "data"):
        if (candidate / "hakaton_analyzed.csv").exists():
            return candidate
    return here / "data"


DATA_DIR = _find_data_dir()
RAW_PATH = DATA_DIR / "hakaton.csv"
CLEAN_PATH = DATA_DIR / "hakaton_analyzed.csv"
ML_PATH = DATA_DIR / "dataset_with_model_scores.csv"


# ----------------------------- ЗАГРУЗКА ----------------------------- #

@st.cache_data(show_spinner="Загружаю исходные данные…")
def load_raw() -> pd.DataFrame:
    """Исходный файл с разделителем ; и названиями школ."""
    df = pd.read_csv(RAW_PATH, sep=";", dtype=str)
    for col in ["test_date", "bdate", "guard_bdate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["child_key"] = _child_key(df)
    return df


@st.cache_data(show_spinner="Загружаю очищенные данные…")
def load_clean() -> pd.DataFrame:
    """Очищенный датасет из analysis.ipynb (hakaton_analyzed.csv)."""
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    for col in ["test_date", "bdate", "guard_bdate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")
    df["variant"] = pd.to_numeric(df["variant"], errors="coerce")

    # Алиасы флагов: имена из analysis.ipynb ---> имена которые ожидает дашборд
    id_numeric = pd.to_numeric(df["id_doc"], errors="coerce")
    df["flag_suspicious_id"] = (id_numeric < 0) | id_numeric.isna()
    df["flag_frequency_violation"] = df["freq_violation"].astype(bool)
    df["flag_age_anomaly"] = (df["child_too_young"] | df["child_too_old"]).astype(bool)
    df["flag_parent_too_young"] = df["guard_too_young_parent"].astype(bool)
    df["flag_parent_child_id_match"] = df["id_equals_guard_id"].astype(bool)

    df["child_key"] = _child_key(df)
    return df


@st.cache_data(show_spinner="Загружаю результаты ML-модели…")
def load_ml_results() -> pd.DataFrame | None:
    """Датасет с оценками Isolation Forest (dataset_with_model_scores.csv)."""
    if not ML_PATH.exists():
        return None
    df = pd.read_csv(ML_PATH, low_memory=False)
    for col in ["test_date", "bdate", "guard_bdate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _child_key(df: pd.DataFrame) -> pd.Series:
    """Составной ключ ребёнка: ФИО + bdate (без id_doc - он ненадёжен)."""
    return (
        df["last_name"].fillna("").str.strip().str.upper() + "|" +
        df["first_name"].fillna("").str.strip().str.upper() + "|" +
        df["middle_name"].fillna("").str.strip().str.upper() + "|" +
        df["bdate"].astype(str)
    )


# ----------------------------- НАРУШЕНИЯ ЧАСТОТЫ ----------------------------- #

MIN_INTERVAL_DAYS = 90  # рекомендация: не чаще 1 раза в 3 месяца


@st.cache_data
def compute_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой записи считает интервал с предыдущим тестом того же ребёнка.
    Возвращает исходный df + колонки prev_date, days_since_prev, is_violation.
    """
    out = df.sort_values(["child_key", "test_date"]).copy()
    out["prev_date"] = out.groupby("child_key")["test_date"].shift(1)
    out["days_since_prev"] = (out["test_date"] - out["prev_date"]).dt.days
    out["is_violation"] = out["days_since_prev"].notna() & (out["days_since_prev"] < MIN_INTERVAL_DAYS)
    return out


@st.cache_data
def violation_summary(df_with_intervals: pd.DataFrame) -> dict:
    v = df_with_intervals[df_with_intervals["is_violation"]]
    return {
        "violation_rows": len(v),
        "violation_children": v["child_key"].nunique(),
        "same_day_rows": int((df_with_intervals["days_since_prev"] == 0).sum()),
        "median_interval_violations": float(v["days_since_prev"].median()) if len(v) else None,
    }


# ----------------------------- АНОМАЛИИ ----------------------------- #

@st.cache_data
def detect_anomalies(clean: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """
    Возвращает словарь {имя_аномалии: DataFrame_подозрительных_строк}.
    Флаги берутся напрямую из hakaton_analyzed.csv (analysis.ipynb).
    """
    out = {}

    # Флаги из analysis.ipynb - порядок соответствует anomalies.md
    flag_map = [
        ("multi_class",            "Один ребёнок - разные классы"),
        ("freq_violation",         "Нарушение частоты (< 90 дней)"),
        ("same_day_test",          "Тест написан несколько раз в один день"),
        ("age_class_mismatch",     "Возраст не соответствует классу"),
        ("id_equals_guard_id",     "id_doc ребёнка совпадает с id родителя"),
        ("id_doc_missing",         "Пустой id_doc"),
        ("guard_id_missing",       "Пустой guard_id_doc"),
        ("variant_invalid",        "Некорректный вариант теста"),
        ("guard_too_young_parent", "Родитель слишком молод (разница < 18 лет)"),
        ("child_too_young",        "Ребёнок слишком юный (< 5 лет)"),
        ("child_too_old",          "Ребёнок слишком старый (> 20 лет)"),
        ("bdate_equals_guard",     "Дата рождения ребёнка совпадает с датой родителя"),
        ("class_invalid",          "Некорректный класс"),
        ("test_before_birth",      "Тест написан до рождения ребёнка"),
        ("ogrn_naprav_bad",        "Некорректный ОГРН направившей организации"),
    ]

    for col, label in flag_map:
        if col in clean.columns:
            out[label] = clean[clean[col].astype(bool)]

    # ОГРН с несколькими названиями - только из исходника (нет флага в clean)
    ogrn_names = raw.groupby("ogrn_naprav")["name_naprav"].transform("nunique")
    out["ОГРН направ. несколько названий"] = (
        raw[ogrn_names > 1][["ogrn_naprav", "name_naprav"]]
        .drop_duplicates().sort_values("ogrn_naprav")
    )
    ogrn_names2 = raw.groupby("ogrn_area")["name_area"].transform("nunique")
    out["ОГРН площадка несколько названий"] = (
        raw[ogrn_names2 > 1][["ogrn_area", "name_area"]]
        .drop_duplicates().sort_values("ogrn_area")
    )

    return out


# ----------------------------- ФОРМАТИРОВАНИЕ ----------------------------- #

def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")
