from __future__ import annotations

import json
import os
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    load_lsi_panel_local,
    load_wide_lsi,
    local_lsi_column,
    lsi_column,
    status_for,
)

st.set_page_config(
    page_title="Аналитик · чат по данным LSI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SMOOTH_WINDOW = 7
LSI_SHIFT = 20.0

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MOD_LABELS = {
    "m1": "M1 · Резервы / RUONIA",
    "m2": "M2 · Репо ЦБ",
    "m3": "M3 · ОФЗ",
    "m4": "M4 · Налоги / сезонность",
    "m5": "M5 · Казначейство",
}
MODULE_DESCRIPTIONS = {
    "m1": (
        "M1 — усреднение обязательных резервов и RUONIA. Стресс растёт, если фактические "
        "остатки на корсчетах сильно превышают обязательные резервы (банки копят "
        "буфер ликвидности под стресс) или RUONIA уходит выше своей нормы (дорогая "
        "овернайт-ликвидность)."
    ),
    "m2": (
        "M2 — аукционы прямого репо ЦБ. Стресс растёт при переспросе (cover ratio > 1, "
        "банки активно занимают у регулятора) или резко положительном спреде ставки "
        "отсечения к ключевой ставке."
    ),
    "m3": (
        "M3 — первичные размещения ОФЗ. Аномально высокий cover ratio = ликвидности "
        "избыток и банки бегут в safe-haven; недоспрос (cover < 1) = локальный дефицит "
        "и нежелание брать длинный риск."
    ),
    "m4": (
        "M4 — налоговый календарь (НДС, прибыль, НДПИ, ЕНП, страховые взносы и т.п.). "
        "Тяжёлые налоговые дни → отток средств клиентов с корсчетов банков. Сигнал "
        "сезонный и предсказуемый, но локально может усиливать стресс."
    ),
    "m5": (
        "M5 — структурный дефицит ликвидности (данные ЦБ) и потоки Казначейства. "
        "Стресс растёт, если структурный дефицит увеличивается (банки в долгу перед "
        "ЦБ) или происходит резкий отток средств Казначейства с корсчетов банков."
    ),
}

_FALLBACK_DRIVERS = [
    {"module": "m1", "score": "m1_shift_mad",                "value": "m1_shift",                 "name": "Спред усреднения резервов",  "units": "млрд руб.",
     "up": "Фактические остатки сильно превышают обязательные — банки копят ликвидность под стресс.",
     "down": "Остатки близки к обязательным — буфера ликвидности нет."},
    {"module": "m1", "score": "m1_ruo_mad",                  "value": "m1_ruo",                   "name": "RUONIA",                    "units": "% годовых",
     "up": "RUONIA выше нормы — стресс межбанка, дорогая овернайт-ликвидность.",
     "down": "RUONIA ниже нормы — избыток ликвидности на межбанке."},
    {"module": "m2", "score": "m2_MAD_score_cover",          "value": "m2_Cover_ratio",           "name": "Cover ratio репо ЦБ",       "units": "×",
     "up": "Переспрос на репо ЦБ — дефицит фондирования.",
     "down": "Низкий спрос на репо — фондирования достаточно."},
    {"module": "m2", "score": "m2_MAD_score_rate_spread",    "value": "m2_Rate_spread",           "name": "Спред репо к ключевой",     "units": "п.п.",
     "up": "Ставка отсечения сильно выше ключевой — банки готовы платить дороже.",
     "down": "Спред около нуля — стресса нет."},
    {"module": "m3", "score": "m3_mad_score_cover",          "value": "m3_cover_ratio",           "name": "Cover ratio ОФЗ",           "units": "×",
     "up": "Переспрос на ОФЗ — банки уходят в safe-haven, профицит ликвидности.",
     "down": "Недоспрос на ОФЗ — банки не готовы вкладываться в длинный долг."},
    {"module": "m4", "score": "_m4_tax_kick",                "value": "m4_tax_event_weight",      "name": "Налоговая нагрузка недели", "units": "",
     "up": "Идёт тяжёлая налоговая неделя — отток средств клиентов на уплату налогов.",
     "down": "Налогового давления нет."},
    {"module": "m5", "score": "m5_MAD_score_liquidity_deficit", "value": "m5_liquidity_deficit",  "name": "Структурный дефицит ЦБ",    "units": "млрд руб.",
     "up": "Дефицит структурной ликвидности — банки в долгу перед ЦБ.",
     "down": "Профицит структурной ликвидности — банки размещают излишки."},
    {"module": "m5", "score": "m5_MAD_score_treasury_pressure", "value": "m5_treasury_pressure",  "name": "Давление казначейства (Δ)", "units": "млрд руб.",
     "up": "Резкий отток средств казначейства — деньги уходят с корсчетов банков.",
     "down": "Приток средств казначейства — ликвидность возвращается в систему."},
]


def _load_drivers() -> list[dict]:
    p = DATA_DIR / "module_drivers.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text())
            drv = obj.get("drivers")
            if isinstance(drv, list) and drv:
                fb = {d["score"]: d for d in _FALLBACK_DRIVERS}
                out = []
                for d in drv:
                    base = fb.get(d.get("score"), {})
                    out.append({**base, **d})
                return out
        except Exception:
            pass
    return _FALLBACK_DRIVERS


DRIVERS = _load_drivers()


MONTHS_RU: dict[int, tuple[str, ...]] = {
    1:  ("январь", "января", "январе"),
    2:  ("февраль", "февраля", "феврале"),
    3:  ("март", "марта", "марте"),
    4:  ("апрель", "апреля", "апреле"),
    5:  ("май", "мая", "мае"),
    6:  ("июнь", "июня", "июне"),
    7:  ("июль", "июля", "июле"),
    8:  ("август", "августа", "августе"),
    9:  ("сентябрь", "сентября", "сентябре"),
    10: ("октябрь", "октября", "октябре"),
    11: ("ноябрь", "ноября", "ноябре"),
    12: ("декабрь", "декабря", "декабре"),
}
_MONTH_TO_NUM: dict[str, int] = {form: n for n, forms in MONTHS_RU.items() for form in forms}
_MONTH_PAT = "|".join(sorted(_MONTH_TO_NUM, key=len, reverse=True))


def parse_period(query: str, data_min: pd.Timestamp, data_max: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, str] | None:
    """Извлекает период из текста запроса. Возвращает (start, end, label) или None.

    Поддерживаются:
      * ISO-дата `YYYY-MM-DD` или `DD.MM.YYYY` (одиночный день → ±7д окно для контекста);
      * `<месяц> <год>` (в любом падеже);
      * чистый год `YYYY`;
      * относительные: «последний год», «последние N дней/месяцев», «за прошлый месяц»;
      * диапазоны «<месяц> – <месяц> <год>», «с <дата> по <дата>».
    Все вычисления опираются на data_max — это «сегодня» с точки зрения данных системы.
    """
    q = query.lower().strip()

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
    if m:
        d = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return _clamp_window(d - timedelta(days=7), d + timedelta(days=7), data_min, data_max,
                             label=f"окно ±7д вокруг {d.strftime('%Y-%m-%d')}")

    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b", q)
    if m:
        d = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return _clamp_window(d - timedelta(days=7), d + timedelta(days=7), data_min, data_max,
                             label=f"окно ±7д вокруг {d.strftime('%Y-%m-%d')}")

    m = re.search(
        rf"(?:с\s+)?({_MONTH_PAT})\s*(?:по|до|-|–|—)\s*({_MONTH_PAT})\s*(\d{{4}})",
        q,
    )
    if m:
        m1, m2, y = _MONTH_TO_NUM[m.group(1)], _MONTH_TO_NUM[m.group(2)], int(m.group(3))
        start = pd.Timestamp(y, m1, 1)
        end = (pd.Timestamp(y, m2, 1) + pd.offsets.MonthEnd(0)).normalize()
        return _clamp_window(start, end, data_min, data_max,
                             label=f"{start.strftime('%b %Y')} … {end.strftime('%b %Y')}")

    m = re.search(r"(?:с\s+)?(\d{4}-\d{2}-\d{2})\s*(?:по|до|-|–|—)\s*(\d{4}-\d{2}-\d{2})", q)
    if m:
        return _clamp_window(pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2)), data_min, data_max,
                             label=f"{m.group(1)} … {m.group(2)}")

    m = re.search(rf"\b({_MONTH_PAT})\s+(\d{{4}})\b", q)
    if m:
        mo, y = _MONTH_TO_NUM[m.group(1)], int(m.group(2))
        start = pd.Timestamp(y, mo, 1)
        end = (start + pd.offsets.MonthEnd(0)).normalize()
        return _clamp_window(start, end, data_min, data_max,
                             label=f"{m.group(1)} {y}")

    m = re.search(rf"\b(\d{{4}})\s+({_MONTH_PAT})\b", q)
    if m:
        y, mo = int(m.group(1)), _MONTH_TO_NUM[m.group(2)]
        start = pd.Timestamp(y, mo, 1)
        end = (start + pd.offsets.MonthEnd(0)).normalize()
        return _clamp_window(start, end, data_min, data_max,
                             label=f"{m.group(2)} {y}")

    if re.search(r"(последн|прошл)\w*\s+год", q) or "за год" in q:
        return _clamp_window(data_max - timedelta(days=365), data_max, data_min, data_max,
                             label="последние 365 дней")
    m = re.search(r"последн\w+\s+(\d+)\s*(дн|дней|месяц|месяца|месяцев)", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        days = n * (30 if unit.startswith("месяц") else 1)
        return _clamp_window(data_max - timedelta(days=days), data_max, data_min, data_max,
                             label=f"последние {n} {'мес' if unit.startswith('месяц') else 'д'}")
    if "прошл" in q and "месяц" in q:
        first_this = data_max.replace(day=1)
        end_prev = first_this - timedelta(days=1)
        start_prev = end_prev.replace(day=1)
        return _clamp_window(start_prev, end_prev, data_min, data_max,
                             label="предыдущий календарный месяц")

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        start = pd.Timestamp(y, 1, 1)
        end = pd.Timestamp(y, 12, 31)
        return _clamp_window(start, end, data_min, data_max, label=f"{y} год")

    return None


def _clamp_window(start: pd.Timestamp, end: pd.Timestamp, dmin: pd.Timestamp, dmax: pd.Timestamp,
                  *, label: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    s = max(pd.Timestamp(start).normalize(), pd.Timestamp(dmin).normalize())
    e = min(pd.Timestamp(end).normalize(), pd.Timestamp(dmax).normalize())
    if s > e:
        return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), label
    return s, e, label


def detect_intent(query: str) -> str:
    """Грубая классификация интента: top-stress / why / period.

    top_stress срабатывает на «пик», «максимум/максимальный», «топ», «худший»,
    «острый/высокий стресс», «наиболее напряжённый» — все встречающиеся в
    запросах из ТЗ формы.
    """
    q = query.lower()
    if re.search(
        r"(\bпик\w*|\bмаксим\w+|\bтоп\b|\bтоп-?\d|\bхудш\w*|"
        r"остр\w+\s+стресс|высок\w+\s+стресс|"
        r"самы\w+\s+(?:стресс|остр|напряж|высок)|"
        r"наиб\w+\s+(?:стресс|остр|напряж|высок|тяж))",
        q,
    ):
        return "top_stress"
    if q.startswith("почему") or " почему " in q or "из-за чего" in q:
        return "why"
    return "period"


def _safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _zone(v: float) -> str:
    if v >= 70: return "🔴 Красный"
    if v >= 40: return "🟡 Жёлтый"
    return "🟢 Зелёный"


def retrieve_period(
    wide: pd.DataFrame,
    panel_global: pd.DataFrame,
    panel_local: pd.DataFrame | None,
    events: pd.DataFrame,
    drivers: list[dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    label: str,
) -> str:
    """Числовой контекст по периоду строго из таблиц."""
    w = wide[(wide["date"] >= start) & (wide["date"] <= end)].copy()
    pg = panel_global[(panel_global["date"] >= start) & (panel_global["date"] <= end)].copy()
    pl = (
        panel_local[(panel_local["date"] >= start) & (panel_local["date"] <= end)].copy()
        if panel_local is not None and not panel_local.empty
        else None
    )

    out: list[str] = []
    out.append(f"# ПЕРИОД: {start.strftime('%Y-%m-%d')} … {end.strftime('%Y-%m-%d')}  ({label})")
    if pg.empty and w.empty:
        out.append("⚠️ В системе НЕТ данных за этот период (за пределами выгрузки).")
        out.append(f"Доступный диапазон данных: {panel_global['date'].min().date()} … {panel_global['date'].max().date()}.")
        return "\n".join(out)

    if not pg.empty:
        s = (_safe(pg["lsi_smooth"]) + LSI_SHIFT).clip(0, 100)
        s_idx = s.reset_index(drop=True)
        n = len(s_idx)
        red = int((s_idx >= 70).sum())
        yel = int(((s_idx >= 40) & (s_idx < 70)).sum())
        grn = int((s_idx < 40).sum())
        i_max = int(s_idx.idxmax())
        i_min = int(s_idx.idxmin())
        out.append("")
        out.append(f"## Глобальный LSI (LSI_lgbm_tuned, шкала 0–100, абсолютная)")
        out.append(f"- среднее = {s_idx.mean():.1f}, медиана = {s_idx.median():.1f}, std = {s_idx.std():.1f}")
        out.append(f"- максимум = {s_idx.iloc[i_max]:.1f} ({pg['date'].iloc[i_max].strftime('%Y-%m-%d')}, {_zone(float(s_idx.iloc[i_max]))})")
        out.append(f"- минимум  = {s_idx.iloc[i_min]:.1f} ({pg['date'].iloc[i_min].strftime('%Y-%m-%d')}, {_zone(float(s_idx.iloc[i_min]))})")
        out.append(f"- начало периода = {s_idx.iloc[0]:.1f}, конец = {s_idx.iloc[-1]:.1f}, изменение Δ = {s_idx.iloc[-1] - s_idx.iloc[0]:+.1f}")
        out.append(f"- дней в красной зоне (LSI ≥ 70): {red}/{n} ({100 * red / n:.0f}%)")
        out.append(f"- дней в жёлтой зоне (40 ≤ LSI < 70): {yel}/{n} ({100 * yel / n:.0f}%)")
        out.append(f"- дней в зелёной зоне (LSI < 40): {grn}/{n} ({100 * grn / n:.0f}%)")

        top = pg.assign(_lsi=s_idx.values).nlargest(5, "_lsi")[["date", "_lsi"]]
        out.append("")
        out.append("### Топ-5 дней по уровню глобального LSI в периоде:")
        for _, r in top.iterrows():
            out.append(f"- {pd.Timestamp(r['date']).strftime('%Y-%m-%d')}: LSI = {r['_lsi']:.1f} ({_zone(float(r['_lsi']))})")

    if pl is not None and not pl.empty:
        ls = _safe(pl["lsi_smooth"]).reset_index(drop=True)
        out.append("")
        out.append("## Локальный LSI (LSI_lgbm_local_multi, перцентиль относительно последних 365 дней)")
        i_max = int(ls.idxmax()); i_min = int(ls.idxmin())
        out.append(f"- среднее = {ls.mean():.1f}, максимум = {ls.iloc[i_max]:.1f} ({pl['date'].iloc[i_max].strftime('%Y-%m-%d')}), минимум = {ls.iloc[i_min]:.1f}")
        out.append(f"- начало = {ls.iloc[0]:.1f}, конец = {ls.iloc[-1]:.1f}, Δ = {ls.iloc[-1] - ls.iloc[0]:+.1f}")
        out.append("  (если глобальный LSI высокий, а локальный — низкий, значит «по истории плохо, но за последние 12 мес. — относительно норма»)")

    if not w.empty:
        out.append("")
        out.append("## Средние сигналы драйверов M1–M5 за период")
        out.append("(MAD-score — робастная z-оценка отклонения от нормы; |MAD| ≥ 1.5 уже заметно)")
        any_drv = False
        for d in drivers:
            sc_col = d["score"]
            if sc_col not in w.columns:
                continue
            sc = _safe(w[sc_col])
            sc_mean = sc.mean(skipna=True)
            sc_max = sc.abs().max(skipna=True)
            if pd.isna(sc_mean) or (abs(float(sc_mean)) < 0.20 and abs(float(sc_max)) < 0.50):
                continue
            any_drv = True
            val_col = d.get("value")
            val_str = ""
            if val_col and val_col in w.columns:
                v_mean = _safe(w[val_col]).mean(skipna=True)
                v_end = _safe(w[val_col]).iloc[-1] if len(w) else None
                if pd.notna(v_mean):
                    val_str = f"; ср. значение = {float(v_mean):.4g} {d['units']}".rstrip()
                if pd.notna(v_end):
                    val_str += f"; на конец периода = {float(v_end):.4g} {d['units']}".rstrip()
            explain = d.get("up") if float(sc_mean) >= 0 else d.get("down")
            out.append(
                f"- [{d['module'].upper()}] {d['name']}: ср.MAD = {float(sc_mean):+.2f}, |MAD|max = {float(sc_max):.2f}{val_str}. {explain or ''}".rstrip()
            )
        if not any_drv:
            out.append("- (значимых отклонений не зафиксировано)")

    if not events.empty:
        evs = events[
            (pd.to_datetime(events["date"]) >= start - pd.Timedelta(days=60))
            & (pd.to_datetime(events["date"]) <= end + pd.Timedelta(days=30))
        ]
        if not evs.empty:
            out.append("")
            out.append("## Стресс-события в окне (−60 / +30 дней от периода)")
            for _, r in evs.sort_values("date").iterrows():
                out.append(f"- {pd.Timestamp(r['date']).strftime('%Y-%m-%d')}: {r['label']}")

    if not w.empty and "m4_tax_event_weight" in w.columns:
        heavy_tax = w[_safe(w["m4_tax_event_weight"]) > 0.6][["date", "m4_tax_event_weight"]].head(8)
        if not heavy_tax.empty:
            out.append("")
            out.append("## Налоговый календарь: дни с высокой нагрузкой (вес > 0.6)")
            for _, r in heavy_tax.iterrows():
                w_val = float(_safe(pd.Series([r["m4_tax_event_weight"]])).iloc[0])
                tax_cols = [
                    c.replace("m4_", "") for c in w.columns
                    if c.startswith("m4_") and c not in {
                        "m4_year", "m4_month", "m4_day", "m4_n_events_html",
                        "m4_n_important", "m4_tax_event_weight", "m4_day_type_code",
                    } and pd.notna(r.get(c)) and float(r.get(c) or 0) > 0
                ]
                tax = ", ".join(tax_cols[:5]) if tax_cols else "налоговый день"
                out.append(f"- {pd.Timestamp(r['date']).strftime('%Y-%m-%d')}: вес = {w_val:.2f} ({tax})")

    return "\n".join(out)


def retrieve_top_stress(
    panel_global: pd.DataFrame,
    wide: pd.DataFrame,
    drivers: list[dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    label: str,
    top_n: int = 6,
) -> str:
    """Поиск эпизодов максимального стресса за период (группировка по близким датам)."""
    pg = panel_global[(panel_global["date"] >= start) & (panel_global["date"] <= end)].copy()
    out: list[str] = [f"# ТОП-ЭПИЗОДЫ СТРЕССА · {start.strftime('%Y-%m-%d')} … {end.strftime('%Y-%m-%d')} ({label})"]
    if pg.empty:
        out.append("⚠️ Нет данных за этот период.")
        return "\n".join(out)

    s = (_safe(pg["lsi_smooth"]) + LSI_SHIFT).clip(0, 100).reset_index(drop=True)
    threshold = max(70.0, float(s.quantile(0.90)))
    out.append(f"Порог острого стресса: LSI ≥ {threshold:.1f} (это либо красная зона ≥70, либо 90-й перцентиль за период).")

    hi_mask = (s >= threshold).to_numpy()
    dates = pd.to_datetime(pg["date"]).reset_index(drop=True)
    if hi_mask.sum() == 0:
        out.append("За этот период LSI ни разу не достиг порога острого стресса.")
        top = pg.assign(_lsi=s.values).nlargest(5, "_lsi")[["date", "_lsi"]]
        out.append("")
        out.append("Топ-5 наиболее напряжённых дней (без формальной красной зоны):")
        for _, r in top.iterrows():
            out.append(f"- {pd.Timestamp(r['date']).strftime('%Y-%m-%d')}: LSI = {r['_lsi']:.1f}")
        return "\n".join(out)

    episodes: list[dict] = []
    in_ep = False
    ep_start = ep_end = None
    ep_peak_val = -1.0
    ep_peak_date = None
    last_date = None
    for i, hi in enumerate(hi_mask):
        d = dates.iloc[i]
        if hi:
            if not in_ep:
                in_ep = True
                ep_start = d
                ep_peak_val = float(s.iloc[i])
                ep_peak_date = d
            else:
                if (d - last_date).days > 7:
                    episodes.append({"start": ep_start, "end": ep_end, "peak_val": ep_peak_val, "peak_date": ep_peak_date})
                    ep_start = d
                    ep_peak_val = float(s.iloc[i])
                    ep_peak_date = d
            if float(s.iloc[i]) > ep_peak_val:
                ep_peak_val = float(s.iloc[i])
                ep_peak_date = d
            ep_end = d
            last_date = d
    if in_ep:
        episodes.append({"start": ep_start, "end": ep_end, "peak_val": ep_peak_val, "peak_date": ep_peak_date})

    episodes.sort(key=lambda ep: ep["peak_val"], reverse=True)
    out.append(f"Обнаружено эпизодов: {len(episodes)} (показываю топ-{min(top_n, len(episodes))}).")

    for i, ep in enumerate(episodes[:top_n], 1):
        ep_days = (ep["end"] - ep["start"]).days + 1
        out.append(f"\n{i}. {ep['start'].strftime('%Y-%m-%d')} … {ep['end'].strftime('%Y-%m-%d')} "
                   f"({ep_days} торг. дней): пик LSI = {ep['peak_val']:.1f} ({ep['peak_date'].strftime('%Y-%m-%d')})")
        row = wide[wide["date"] == ep["peak_date"]]
        if not row.empty:
            r = row.iloc[0]
            top_drv: list[tuple] = []
            for d in drivers:
                sc_col = d["score"]
                if sc_col not in row.columns:
                    continue
                sc = pd.to_numeric(pd.Series([r.get(sc_col)]), errors="coerce").iloc[0]
                if pd.isna(sc) or abs(float(sc)) < 0.5:
                    continue
                val = r.get(d.get("value"))
                val_str = ""
                if d.get("value") and pd.notna(val):
                    val_str = f", значение = {float(val):.4g} {d['units']}".rstrip()
                top_drv.append((abs(float(sc)), f"   • [{d['module'].upper()}] {d['name']}: MAD = {float(sc):+.2f}{val_str}"))
            top_drv.sort(key=lambda t: t[0], reverse=True)
            for _, line in top_drv[:3]:
                out.append(line)
            if not top_drv:
                out.append("   • (драйверы M1–M5 в этот день без значимых отклонений)")
    return "\n".join(out)


def retrieve_overview(
    wide: pd.DataFrame,
    panel_global: pd.DataFrame,
    panel_local: pd.DataFrame | None,
) -> str:
    """Сводка по системе, если запрос без явного периода (для onboarding)."""
    if panel_global.empty:
        return "В системе нет данных LSI."
    s = (_safe(panel_global["lsi_smooth"]) + LSI_SHIFT).clip(0, 100)
    latest = float(s.iloc[-1])
    last_date = pd.Timestamp(panel_global["date"].iloc[-1])
    out = [
        "# ОБЩАЯ СВОДКА ПО СИСТЕМЕ LSI",
        f"- Доступный диапазон данных: {panel_global['date'].min().date()} … {panel_global['date'].max().date()} ({len(panel_global)} торговых дней).",
        f"- Последняя точка: {last_date.strftime('%Y-%m-%d')}, глобальный LSI = {latest:.1f}/100 ({_zone(latest)}).",
    ]
    if panel_local is not None and not panel_local.empty:
        ls = _safe(panel_local["lsi_smooth"]).iloc[-1]
        if pd.notna(ls):
            out.append(f"- Локальный LSI на ту же дату = {float(ls):.1f}/100 ({_zone(float(ls))}).")
    threshold = max(70.0, float(s.quantile(0.95)))
    out.append(f"- Топ-3 эпизода острого стресса (LSI ≥ {threshold:.1f}) за всё время — см. ниже.")
    out.append("")
    out.append(retrieve_top_stress(
        panel_global, wide, DRIVERS,
        pd.Timestamp(panel_global["date"].min()),
        pd.Timestamp(panel_global["date"].max()),
        label="вся история", top_n=3,
    ))
    return "\n".join(out)


def build_modules_block() -> str:
    """Статический блок: описания модулей (часть RAG-знаний)."""
    lines = ["# ОПИСАНИЕ МОДУЛЕЙ СИСТЕМЫ (статика)"]
    for code in ("m1", "m2", "m3", "m4", "m5"):
        lines.append(f"- **{MOD_LABELS[code]}**: {MODULE_DESCRIPTIONS[code]}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "Ты — старший аналитик казначейства банка ПСБ, отвечающий за систему LSI "
    "(Liquidity Stress Index) рублёвого денежного рынка. Ты ведёшь диалог с пользователем "
    "дашборда. У тебя есть ДОСТУП ТОЛЬКО к числовым данным системы, которые передаются в "
    "блоке <DATA_CONTEXT>. У тебя НЕТ интернета и НЕТ другой информации.\n\n"
    "СТРОГИЕ ПРАВИЛА:\n"
    "1) Любая цифра, дата, факт в твоём ответе должны быть взяты из <DATA_CONTEXT>. "
    "Категорически нельзя выдумывать значения, события, ставки ЦБ, новости.\n"
    "2) Если пользователь спрашивает про период / событие, по которому в <DATA_CONTEXT> "
    "сказано «нет данных» или такого блока нет — честно отвечай: «В системе нет данных "
    "за этот период / по этому вопросу».\n"
    "3) Ссылайся на конкретные числа и даты из <DATA_CONTEXT> (например: «LSI достиг "
    "92.8 на 2022-02-28»). Не используй размытых формулировок «примерно высокий».\n"
    "4) Используй описания модулей M1–M5 из <MODULES> для интерпретации сигналов, но "
    "сами числа — только из <DATA_CONTEXT>.\n"
    "5) Тон: краткий, профессиональный, на русском. Не более 5–6 абзацев на ответ.\n"
    "6) Если пользователь уточняет («а что в апреле?»), используй контекст этого хода + "
    "помни, о чём говорили раньше в этом чате.\n"
)


def _get_api_key() -> str:
    return (
        st.session_state.get("deepseek_key")
        or os.getenv("DEEPSEEK_API_KEY", "")
    )


def call_llm(messages: list[dict]) -> str | None:
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        rsp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return rsp.choices[0].message.content
    except Exception as e:
        st.error(f"LLM ошибка: {e}")
        return None


@st.cache_data(show_spinner=False)
def _load_panel_global() -> pd.DataFrame:
    p = DATA_DIR / "lsi_panel.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
    else:
        wide_local = load_wide_lsi()
        col = lsi_column(wide_local)
        df = pd.DataFrame({"date": wide_local["date"], "lsi_raw": wide_local[col].clip(0, 100)})
        df["lsi_smooth"] = df["lsi_raw"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_panel_local() -> pd.DataFrame | None:
    pl = load_lsi_panel_local()
    if pl is not None and not pl.empty:
        return pl
    wide_local = load_wide_lsi()
    col = local_lsi_column(wide_local)
    if col is None:
        return None
    out = wide_local[["date", col]].dropna(subset=[col]).copy()
    out["lsi_raw"] = out[col].clip(0, 100)
    out["lsi_smooth"] = out["lsi_raw"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_events() -> pd.DataFrame:
    p = DATA_DIR / "stress_events.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "label"])


wide = load_wide_lsi().sort_values("date").reset_index(drop=True)
panel_global = _load_panel_global()
panel_local = _load_panel_local()
events = _load_events()

data_min = pd.Timestamp(panel_global["date"].min())
data_max = pd.Timestamp(panel_global["date"].max())


st.markdown(
    """
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
.chat-hero {
    display:flex; align-items:center; gap:18px;
    padding:18px 22px; border-radius:14px;
    background: linear-gradient(135deg, rgba(31,78,121,0.08), rgba(10,107,61,0.04));
    border:1px solid rgba(127,127,127,0.18); margin-bottom:8px;
}
.chat-muted { opacity: 0.72; font-size: 0.9rem; }
.example-q-btn button { font-size: 13px !important; padding: 4px 10px !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="chat-hero">
  <div style="flex:1">
    <div style="font-size:1.55rem; font-weight:700; margin:0">🤖 Аналитик · чат по данным LSI</div>
    <div class="chat-muted" style="margin-top:4px">
      RAG строго по таблицам системы. LLM видит только: исторические LSI, сигналы M1–M5,
      налоговый календарь, стресс-события и описания модулей. Никакой внешней информации.
    </div>
  </div>
  <div style="text-align:right; opacity:.8; font-size:13px">
    Данные: <b>{data_min.strftime('%Y-%m-%d')}</b> … <b>{data_max.strftime('%Y-%m-%d')}</b><br/>
    {len(panel_global)} торговых дней
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Настройки")
    st.session_state["deepseek_key"] = st.text_input(
        "DeepSeek API key",
        type="password",
        value=_get_api_key(),
        placeholder="sk-…",
        help="Ключ DeepSeek. Можно задать через env var DEEPSEEK_API_KEY. "
             "Без ключа ответ LLM будет недоступен — но контекст RAG всё равно покажется.",
    )
    if _get_api_key():
        st.success("Ключ задан — чат активен", icon="✅")
    else:
        st.warning("Без API-ключа LLM не отвечает", icon="⚠️")

    if st.button("🗑 Очистить историю", use_container_width=True):
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("chat_last_period", None)
        st.rerun()

    st.divider()
    st.caption(
        "ИСТОЧНИКИ ДАННЫХ (RAG-источники):\n"
        "• `wide_lsi.csv` — сигналы и флаги M1–M5\n"
        "• `lsi_panel.csv` — глобальный LSI\n"
        "• `lsi_panel_local.csv` — локальный LSI (если есть)\n"
        "• `stress_events.csv` — стресс-события\n"
        "• описания модулей (статика)\n"
    )

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

st.markdown("**Примеры вопросов** (клик — задаст вопрос):")
cols = st.columns(3)
examples = [
    "Что происходило с ликвидностью в марте 2022?",
    "Почему в августе 2023 вырос LSI?",
    "Покажи периоды максимального стресса за последний год",
]
queued_query = st.session_state.pop("queued_query", None)
for i, q in enumerate(examples):
    if cols[i].button(q, key=f"ex_{i}", use_container_width=True):
        queued_query = q

st.markdown("---")

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        ctx = msg.get("context")
        if ctx:
            with st.expander("🔍 Сырой контекст RAG (что видела LLM)"):
                st.code(ctx, language="markdown")

user_query = st.chat_input("Спросите про любой период, событие или модуль…")
if queued_query and not user_query:
    user_query = queued_query

if user_query:
    st.session_state["chat_messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    period = parse_period(user_query, data_min, data_max)
    if period is None:
        period = st.session_state.get("chat_last_period")
    intent = detect_intent(user_query)

    ctx_parts: list[str] = []
    ctx_parts.append(build_modules_block())

    if period is not None:
        start, end, label = period
        if intent == "top_stress":
            ctx_parts.append(retrieve_top_stress(panel_global, wide, DRIVERS, start, end, label=label))
        else:
            ctx_parts.append(retrieve_period(wide, panel_global, panel_local, events, DRIVERS, start, end, label=label))
        st.session_state["chat_last_period"] = period
    else:
        if intent == "top_stress":
            ctx_parts.append(retrieve_top_stress(
                panel_global, wide, DRIVERS,
                pd.Timestamp(panel_global["date"].min()),
                pd.Timestamp(panel_global["date"].max()),
                label="вся история", top_n=6,
            ))
        else:
            ctx_parts.append(retrieve_overview(wide, panel_global, panel_local))

    context_text = "\n\n".join(ctx_parts)

    HISTORY_TURNS = 6
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    hist = st.session_state["chat_messages"][:-1]
    for m in hist[-2 * HISTORY_TURNS:]:
        messages.append({"role": m["role"], "content": m["content"]})

    enriched_user = (
        f"Вопрос пользователя: {user_query}\n\n"
        f"<MODULES>\n{ctx_parts[0]}\n</MODULES>\n\n"
        f"<DATA_CONTEXT>\n{ctx_parts[1] if len(ctx_parts) > 1 else '(нет данных)'}\n</DATA_CONTEXT>\n\n"
        f"Ответь на вопрос, используя ТОЛЬКО числа и даты из <DATA_CONTEXT> и интерпретацию из <MODULES>."
    )
    messages.append({"role": "user", "content": enriched_user})

    with st.chat_message("assistant"):
        if not _get_api_key():
            answer = (
                "⚠️ API-ключ DeepSeek не задан — LLM-разбор недоступен. "
                "Но вот сырой RAG-контекст, собранный по таблицам системы для вашего запроса:\n\n"
                "```markdown\n" + context_text + "\n```"
            )
            st.markdown(answer)
        else:
            with st.spinner("DeepSeek анализирует контекст…"):
                raw = call_llm(messages)
            if raw is None:
                answer = (
                    "Не удалось получить ответ от LLM. Проверьте API-ключ. "
                    "Контекст RAG (что было бы отдано модели) — в экспандере ниже."
                )
            else:
                answer = raw
            st.markdown(answer)
        with st.expander("🔍 Сырой контекст RAG (что видела LLM)"):
            st.code(context_text, language="markdown")

    st.session_state["chat_messages"].append({
        "role": "assistant",
        "content": answer,
        "context": context_text,
    })
    st.rerun()
