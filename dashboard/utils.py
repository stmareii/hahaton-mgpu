"""
Общие функции для дашборда: загрузка данных, нормализация, детекторы аномалий.
Всё закэшировано через st.cache_data, так что страницы работают быстро.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

def _find_data_dir() -> Path:
    """
    Ищем data/ сначала рядом с utils.py (dashboard/data/),
    потом на уровень выше (корень проекта: hahaton-mgpu/data/).
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / "data", here.parent / "data"):
        if (candidate / "without_duplicates.csv").exists():
            return candidate
    # ничего не нашли - вернём дефолт, pandas сам покажет понятную ошибку
    return here / "data"


DATA_DIR = _find_data_dir()
RAW_PATH = DATA_DIR / "hakaton.csv"
CLEAN_PATH = DATA_DIR / "without_duplicates.csv"


# ----------------------------- ЗАГРУЗКА ----------------------------- #

@st.cache_data(show_spinner="Загружаю исходные данные…")
def load_raw() -> pd.DataFrame:
    """Исходный файл с разделителем ; и названиями школ."""
    df = pd.read_csv(RAW_PATH, sep=";", dtype=str)
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    df["bdate"] = pd.to_datetime(df["bdate"], errors="coerce")
    df["guard_bdate"] = pd.to_datetime(df["guard_bdate"], errors="coerce")
    df["child_key"] = _child_key(df)
    return df


@st.cache_data(show_spinner="Загружаю очищенные данные…")
def load_clean() -> pd.DataFrame:
    """Очищенный датасет сокомандника (без дубликатов, без названий школ)."""
    df = pd.read_csv(CLEAN_PATH, dtype=str)
    # лишний индексный столбец после to_csv без index=False
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    df["bdate"] = pd.to_datetime(df["bdate"], errors="coerce")
    df["guard_bdate"] = pd.to_datetime(df["guard_bdate"], errors="coerce")
    df["child_key"] = _child_key(df)
    return df


def _child_key(df: pd.DataFrame) -> pd.Series:
    """Единый ключ ребёнка: ФИО + bdate + id_doc (рекомендация из EDA)."""
    return (
        df["last_name"].fillna("").str.strip().str.upper() + "|" +
        df["first_name"].fillna("").str.strip().str.upper() + "|" +
        df["bdate"].astype(str) + "|" +
        df["id_doc"].fillna("").str.strip()
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

VALID_CLASSES = {str(i) for i in range(1, 12)}


@st.cache_data
def detect_anomalies(clean: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """
    Возвращает словарь {имя_аномалии: DataFrame_подозрительных_строк}.
    Каждую можно показать в таблице с кнопкой выгрузки.
    """
    out = {}

    # 1. id_doc с минусом
    out["id_doc_отрицательный"] = clean[clean["id_doc"].fillna("").str.startswith("-")]

    # 2. Слишком короткий id_doc
    short = clean["id_doc"].fillna("").str.len().between(1, 5)
    out["id_doc_слишком_короткий"] = clean[short]

    # 3. Дубли (ребёнок, дата) - только в исходнике
    dup_mask = raw.duplicated(subset=["child_key", "test_date"], keep=False)
    out["дубли_тестов_в_один_день"] = raw[dup_mask].sort_values(["child_key", "test_date"])

    # 4. ОГРН с несколькими названиями школ
    ogrn_names = raw.groupby("ogrn_naprav")["name_naprav"].transform("nunique")
    out["ОГРН_направ_несколько_названий"] = (
        raw[ogrn_names > 1][["ogrn_naprav", "name_naprav"]]
        .drop_duplicates()
        .sort_values("ogrn_naprav")
    )
    ogrn_names2 = raw.groupby("ogrn_area")["name_area"].transform("nunique")
    out["ОГРН_площадка_несколько_названий"] = (
        raw[ogrn_names2 > 1][["ogrn_area", "name_area"]]
        .drop_duplicates()
        .sort_values("ogrn_area")
    )

    # 5. id_doc у разных ФИО
    valid = clean[clean["id_doc"].notna() & (clean["id_doc"] != "-")]
    fio_per_id = valid.groupby("id_doc")[["last_name", "first_name", "bdate"]] \
                      .transform(lambda s: s.astype(str))
    key_by_id = valid.groupby("id_doc").apply(
        lambda g: g[["last_name", "first_name", "bdate"]].drop_duplicates().shape[0]
    )
    bad_ids = key_by_id[key_by_id > 1].index
    out["один_id_doc_разные_ФИО"] = valid[valid["id_doc"].isin(bad_ids)] \
        .sort_values(["id_doc", "last_name"])

    # 6. Более 2 опекунов
    guards = clean.groupby("child_key").apply(
        lambda g: g[["guard_last_name", "guard_first_name", "guard_id_doc"]].drop_duplicates().shape[0]
    )
    many_guards_keys = guards[guards > 2].index
    out["больше_2_опекунов"] = clean[clean["child_key"].isin(many_guards_keys)] \
        .sort_values("child_key")

    # 7. Некорректный класс
    out["некорректный_класс"] = clean[~clean["class"].isin(VALID_CLASSES)]

    # 8. Возраст не соответствует классу
    age = (clean["test_date"] - clean["bdate"]).dt.days / 365.25
    def class_ok(row_class, row_age):
        try:
            c = int(row_class)
        except (ValueError, TypeError):
            return True  # некорректный класс отлавливается отдельно
        if pd.isna(row_age):
            return True
        return (c + 4) <= row_age <= (c + 9)  # с запасом ±1 год

    ok_mask = np.array([class_ok(c, a) for c, a in zip(clean["class"], age)])
    tmp = clean.copy()
    tmp["age_at_test"] = age.round(1)
    out["возраст_не_соответствует_классу"] = tmp[~ok_mask]

    # 9. Опекун моложе ребёнка
    out["опекун_моложе_ребёнка"] = clean[
        clean["guard_bdate"].notna()
        & clean["bdate"].notna()
        & (clean["guard_bdate"] >= clean["bdate"])
    ]

    # 10. Разница возраста опекун-ребёнок < 14 лет (нереалистично для родителя)
    age_diff = (clean["bdate"] - clean["guard_bdate"]).dt.days / 365.25
    out["опекун_слишком_молод"] = clean[(age_diff > 0) & (age_diff < 14)]

    return out


# ----------------------------- ФОРМАТИРОВАНИЕ ----------------------------- #

def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")
