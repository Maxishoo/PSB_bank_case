from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"

COLOR_GREEN = "#1f9d55"
COLOR_YELLOW = "#e0a800"
COLOR_RED = "#c82333"
COLOR_NEUTRAL = "#5c6770"

LSI_THRESHOLDS = (40.0, 70.0)

LSI_SMOOTH_WINDOW = 7
LSI_DISPLAY_SHIFT = 20.0


@st.cache_data(show_spinner=False)
def load_wide_lsi() -> pd.DataFrame:
    """Читает экспортированный wide_lsi (CSV/parquet) и приводит date к datetime."""
    parquet = DATA_DIR / "wide_lsi.parquet"
    csv = DATA_DIR / "wide_lsi.csv"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            "Не найден файл данных. Экспортируйте `wide_lsi` из final.ipynb в "
            f"{csv} (см. README)."
        )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df


_FFILL_EXCLUDE: frozenset[str] = frozenset({
    "m1_Flag_EndOfPeriod",
    "m2_Flag_Demand",
    "m3_flag_nedospros", "m3_flag_perespros",
    "m4_tax_event_weight", "_m4_tax_kick", "m4_year", "m4_month",
    "m4_day", "m4_day_type_code", "m4_n_events_html", "m4_n_important",
    "m5_is_month_end", "m5_flag_stress",
})
_FFILL_PREFIXES: tuple[str, ...] = ("m1_", "m2_", "m3_", "m5_")


def _mad_score_series(series: pd.Series, window: int = 756) -> pd.Series:
    """Робастный z-score: (x − rolling_median) / (1.4826 × rolling_MAD)."""
    s = pd.to_numeric(series, errors="coerce")
    min_p = max(30, window // 10)
    med = s.rolling(window, min_periods=min_p).median()
    mad = (s - med).abs().rolling(window, min_periods=min_p).median()
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)


def _ffill_continuous_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill M1/M2/M3/M5 через выходные; m4 и флаги не трогаем."""
    out = df.copy()
    cols = [
        c for c in out.columns
        if any(c.startswith(p) for p in _FFILL_PREFIXES) and c not in _FFILL_EXCLUDE
    ]
    if cols:
        out[cols] = out[cols].ffill().bfill()
    return out


def _normalize_wide_units(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Ключевая ставка m2 в %, пересчёт m2_Rate_spread и MAD m2."""
    df = wide_df.copy()
    if "m2_Ключевая_ставка" in df.columns and "m2_Ставка_отсечения" in df.columns:
        kr = pd.to_numeric(df["m2_Ключевая_ставка"], errors="coerce")
        if kr.dropna().max() > 50:
            df["m2_Ключевая_ставка"] = kr / 100.0
        df["m2_Rate_spread"] = (
            pd.to_numeric(df["m2_Ставка_отсечения"], errors="coerce")
            - pd.to_numeric(df["m2_Ключевая_ставка"], errors="coerce")
        )
        for src, dst in [
            ("m2_Cover_ratio", "m2_MAD_score_cover"),
            ("m2_Rate_spread", "m2_MAD_score_rate_spread"),
        ]:
            if src in df.columns and dst in df.columns:
                mask = df[src].notna()
                if mask.sum() > 30:
                    z = _mad_score_series(df.loc[mask, src], window=156)
                    df.loc[mask, dst] = z.values
                    df[dst] = df[dst].fillna(0.0)
    return df


def aligned_shifted_lsi_series_for_wide(wide: pd.DataFrame) -> pd.Series:
    """Сглаженный глобальный LSI + калибровочный сдвиг, по одному значению на строку `wide`.

    Логика совпадает с главной страницей при наличии `lsi_panel.csv`: сглаживание
    считается из `lsi_raw` (окно ``LSI_SMOOTH_WINDOW``), а не из готовой колонки
    `lsi_smooth` в файле — в выгрузке она может отличаться от актуальной формулы.

    Ряд выравнивается по ``wide["date"]`` (reindex + ffill/bfill), чтобы длина
    совпадала с подготовленным wide и не оставалось «висячих» NaN на стыке календарей.
    """
    if wide.empty or "date" not in wide.columns:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(wide["date"], errors="coerce").dt.normalize()
    panel_path = DATA_DIR / "lsi_panel.csv"
    if panel_path.exists():
        p = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date")
        p_dates = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
        if "lsi_raw" in p.columns:
            raw = pd.to_numeric(p["lsi_raw"], errors="coerce")
            smooth = raw.rolling(LSI_SMOOTH_WINDOW, min_periods=1, center=True).mean()
        elif "lsi_smooth" in p.columns:
            smooth = pd.to_numeric(p["lsi_smooth"], errors="coerce")
        else:
            raise ValueError("lsi_panel.csv: ожидаются колонки lsi_raw или lsi_smooth")
        s_panel = pd.Series(smooth.to_numpy(dtype=float), index=pd.DatetimeIndex(p_dates))
        aligned = s_panel.reindex(idx).ffill().bfill()
    else:
        col = lsi_column(wide)
        raw = pd.to_numeric(wide[col], errors="coerce").clip(0, 100)
        smooth = raw.rolling(LSI_SMOOTH_WINDOW, min_periods=1, center=True).mean()
        aligned = pd.Series(smooth.to_numpy(dtype=float), index=idx)
    out = (aligned + LSI_DISPLAY_SHIFT).clip(0, 100)
    out = pd.to_numeric(out, errors="coerce")
    if out.isna().any():
        out = out.ffill().bfill()
    return out.reset_index(drop=True)


def prepare_wide_lsi_for_charts(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Сортировка дат, нормализация m2, ffill непрерывных рядов — линии без «дыр» по выходным."""
    if df is None:
        out = load_wide_lsi()
    else:
        out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out = _normalize_wide_units(out)
    out = _ffill_continuous_drivers(out)
    return out


def synthetic_ohlc_from_close(close: pd.Series, dates: pd.Series) -> pd.DataFrame | None:
    """Строит упрощённые OHLC из одного ряда (для японских свечей без intraday).

    Для каждой точки: *open* — значение предыдущего дня (у первой точки open = close),
    *high* / *low* — max/min(open, close). Пропуски в ряду отбрасываются.
    """
    df = pd.DataFrame(
        {"dt": pd.to_datetime(dates, errors="coerce"), "c": pd.to_numeric(close, errors="coerce")}
    )
    df = df.dropna(subset=["c", "dt"])
    if df.empty:
        return None
    o = df["c"].shift(1)
    o.iloc[0] = df["c"].iloc[0]
    df = df.assign(o=o)
    df["h"] = df[["o", "c"]].max(axis=1)
    df["l"] = df[["o", "c"]].min(axis=1)
    return df


def lsi_column(df: pd.DataFrame) -> str:
    """Выбирает доступную колонку **глобального** LSI.

    Согласован с тем, что экспорт из ноутбука использует `LSI_lgbm_tuned`
    как итоговый продукт (см. final.ipynb → «Экспорт данных для дашборда»).
    Остальные варианты остаются как fallback на случай старых выгрузок.
    """
    for c in (
        "LSI_lgbm_tuned",
        "LSI_lgbm_tuned_tax_adj",
        "LSI_ensemble",
        "LSI_ensemble_tax_adj",
        "LSI_lgbm_huber",
        "LSI_teacher",
    ):
        if c in df.columns:
            return c
    raise KeyError("В данных нет ни одной колонки LSI_*.")


def local_lsi_column(df: pd.DataFrame) -> str | None:
    """Колонка локального (multi-window) LSI, если присутствует.

    Локальный LSI отвечает на вопрос «как сейчас относительно последнего
    года», тогда как глобальный — на «как сейчас относительно всей истории».
    """
    for c in ("LSI_lgbm_local_multi", "LSI_lgbm_local"):
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_lsi_panel_local() -> pd.DataFrame | None:
    """Загружает локальную LSI-панель (lsi_panel_local.csv).

    Возвращает None, если файл ещё не выгружен — дашборд тогда строит ряд на
    лету из `wide_lsi[LSI_lgbm_local_multi]`.
    """
    p = DATA_DIR / "lsi_panel_local.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def status_for(value: float) -> tuple[str, str, str]:
    """Возвращает (status, emoji, color) по значению LSI."""
    if value is None or not np.isfinite(value):
        return "нет данных", "⚪", COLOR_NEUTRAL
    lo, hi = LSI_THRESHOLDS
    if value < lo:
        return "Зелёный", "🟢", COLOR_GREEN
    if value < hi:
        return "Жёлтый", "🟡", COLOR_YELLOW
    return "Красный", "🔴", COLOR_RED


def status_band_shapes() -> list[dict]:
    """Горизонтальные полосы алертов для plotly (для y∈[0,100])."""
    lo, hi = LSI_THRESHOLDS
    return [
        dict(type="rect", xref="paper", x0=0, x1=1, y0=0, y1=lo,
             fillcolor=COLOR_GREEN, opacity=0.06, line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, y0=lo, y1=hi,
             fillcolor=COLOR_YELLOW, opacity=0.07, line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, y0=hi, y1=100,
             fillcolor=COLOR_RED, opacity=0.07, line_width=0, layer="below"),
    ]


def filter_window(df: pd.DataFrame, dates: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Фильтрация по диапазону дат с защитой от tz-naive."""
    a, b = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    return df[(df["date"] >= a) & (df["date"] <= b)].copy()


def first_present(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def latest_row(df: pd.DataFrame) -> pd.Series | None:
    """Последняя строка с непустым LSI."""
    if df.empty:
        return None
    col = lsi_column(df)
    sub = df[df[col].notna()]
    if sub.empty:
        return None
    return sub.iloc[-1]


def module_share(df: pd.DataFrame) -> pd.DataFrame:
    """Грубая оценка вклада модулей в текущий LSI: норма последних значений MAD-фич."""
    cands = {
        "M1 — Усреднение резервов": ("m1_shift_mad", "m1_ruo_mad"),
        "M2 — Репо ЦБ":             ("m2_MAD_score_cover", "m2_MAD_score_rate_spread"),
        "M3 — ОФЗ":                 ("m3_mad_score_cover",),
        "M4 — Налоги (post-hoc)":   ("_m4_tax_kick",),
        "M5 — Казначейство":        ("m5_MAD_score_treasury_pressure", "m5_MAD_score_liquidity_deficit"),
    }
    if df.empty:
        return pd.DataFrame(columns=["module", "score"])
    last = df.iloc[-1]
    rows = []
    for name, cols in cands.items():
        vals = [abs(float(last[c])) for c in cols if c in df.columns and pd.notna(last[c])]
        if not vals:
            continue
        rows.append({"module": name, "score": float(np.mean(vals))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["share"] = out["score"] / out["score"].sum() if out["score"].sum() > 0 else 0.0
    return out.sort_values("share", ascending=False).reset_index(drop=True)
