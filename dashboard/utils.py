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
        if (candidate / "cleaned.csv").exists():
            return candidate
    # ничего не нашли - вернём дефолт, pandas сам покажет понятную ошибку
    return here / "data"


DATA_DIR = _find_data_dir()
RAW_PATH = DATA_DIR / "hakaton.csv"
CLEAN_PATH = DATA_DIR / "cleaned.csv"


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
    """Очищенный датасет из prepare_data.ipynb (cleaned.csv)."""
    df = pd.read_csv(CLEAN_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    for col in ["test_date", "bdate", "guard_bdate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")
    df["variant"] = pd.to_numeric(df["variant"], errors="coerce")
    for col in [c for c in df.columns if c.startswith("flag_")]:
        df[col] = df[col].astype(bool)
    # child_key уже сохранён в CSV — используем его напрямую
    return df


def _child_key(df: pd.DataFrame) -> pd.Series:
    """Составной ключ ребёнка: ФИО + bdate (без id_doc — он ненадёжен)."""
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
    Каждую можно показать в таблице с кнопкой выгрузки.
    """
    out = {}

    # 1. Подозрительный id_doc (флаг из cleaned.csv: отрицательный или нечисловой)
    out["id_doc подозрительный"] = clean[clean["flag_suspicious_id"]]

    # 2. id_doc ребёнка совпадает с id опекуна (флаг из cleaned.csv)
    out["id_doc ребёнка совпадает с опекуном"] = clean[clean["flag_parent_child_id_match"]]

    # 3. Дубли (ребёнок, дата) — только в исходнике
    dup_mask = raw.duplicated(subset=["child_key", "test_date"], keep=False)
    out["Дубли тестов в один день"] = raw[dup_mask].sort_values(["child_key", "test_date"])

    # 4. ОГРН с несколькими названиями школ
    ogrn_names = raw.groupby("ogrn_naprav")["name_naprav"].transform("nunique")
    out["ОГРН направ. несколько названий"] = (
        raw[ogrn_names > 1][["ogrn_naprav", "name_naprav"]]
        .drop_duplicates()
        .sort_values("ogrn_naprav")
    )
    ogrn_names2 = raw.groupby("ogrn_area")["name_area"].transform("nunique")
    out["ОГРН площадка несколько названий"] = (
        raw[ogrn_names2 > 1][["ogrn_area", "name_area"]]
        .drop_duplicates()
        .sort_values("ogrn_area")
    )

    # 5. id_doc у разных ФИО
    valid = clean[clean["id_doc"].notna()]
    key_by_id = valid.groupby("id_doc").apply(
        lambda g: g[["last_name", "first_name", "bdate"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    bad_ids = key_by_id[key_by_id > 1].index
    out["Один id_doc разные ФИО"] = valid[valid["id_doc"].isin(bad_ids)] \
        .sort_values(["id_doc", "last_name"])

    # 6. Более 2 опекунов
    guards = clean.groupby("child_key").apply(
        lambda g: g[["guard_last_name", "guard_first_name", "guard_id_doc"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    many_guards_keys = guards[guards > 2].index
    out["Больше 2-ух опекунов"] = clean[clean["child_key"].isin(many_guards_keys)] \
        .sort_values("child_key")

    # 7. Возраст аномальный (флаг из cleaned.csv: < 6 или > 18 лет)
    out["Аномальный возраст"] = clean[clean["flag_age_anomaly"]]

    # 8. Возраст не соответствует классу
    age = clean["age_at_test"]
    def class_ok(row_class, row_age):
        try:
            c = int(row_class)
        except (ValueError, TypeError):
            return True
        if pd.isna(row_age):
            return True
        return (c + 4) <= row_age <= (c + 9)

    ok_mask = np.array([class_ok(c, a) for c, a in zip(clean["class"], age)])
    out["Возраст не соответствует классу"] = clean[~ok_mask]

    # 9. Опекун моложе ребёнка
    out["Опекун моложе ребёнка"] = clean[
        clean["guard_bdate"].notna()
        & clean["bdate"].notna()
        & (clean["guard_bdate"] >= clean["bdate"])
    ]

    # 10. Опекун слишком молод (флаг из cleaned.csv: разница < 14 лет)
    out["Опекун слишком молод"] = clean[clean["flag_parent_too_young"]]

    return out


# ----------------------------- ФОРМАТИРОВАНИЕ ----------------------------- #

def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")
