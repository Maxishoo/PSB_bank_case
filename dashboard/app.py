from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from glossary import (
    STAT_LABELS_FOR_LLM,
    TERMS,
    inline_hints,
    llm_lexicon_block,
    metric_help,
    module_label_html,
    render_glossary_llm_expander,
    term_abbr,
)
from utils import (
    load_lsi_panel_local,
    load_wide_lsi,
    local_lsi_column,
    lsi_column,
    prepare_wide_lsi_for_charts,
    status_band_shapes,
    status_for,
    synthetic_ohlc_from_close,
)

st.set_page_config(
    page_title="LSI · Liquidity Stress Index",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).resolve().parent / "data"

LSI_SHIFT = 20.0
HIST_WINDOW = 252
SMOOTH_WINDOW = 7

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY_DEFAULT = ""

MOD_LABELS = {
    "m1": "M1 · Резервы / RUONIA",
    "m2": "M2 · Репо ЦБ",
    "m3": "M3 · ОФЗ",
    "m4": "M4 · Налоги / сезонность",
    "m5": "M5 · Казначейство",
}
MOD_COLORS = {
    "m1": "#1f4e79",
    "m2": "#0a6b3d",
    "m3": "#8e44ad",
    "m4": "#d35400",
    "m5": "#7f6b2c",
}
STATUS_COLORS = {"🟢": "#1e8a4f", "🟡": "#c98a17", "🔴": "#b03a2e"}


st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] > .main .block-container,
section.main > div.block-container {
    padding-top: max(5.75rem, calc(3rem + env(safe-area-inset-top, 0px)));
    padding-bottom: 2rem;
    max-width: 1400px;
}
div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="column"]:last-child {
    flex: 1.15 1 auto !important;
    min-width: 10rem;
    align-self: flex-start;
}
h1, h2, h3, h4 { letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }

.lsi-hero {
    display: flex; align-items: center; gap: 18px;
    padding: 18px 22px 18px 18px; border-radius: 14px;
    background: linear-gradient(135deg, rgba(31,78,121,0.08), rgba(10,107,61,0.04));
    border: 1px solid rgba(127,127,127,0.18);
    margin: 0 0 12px 0;
    width: 100%;
    box-sizing: border-box;
}
.lsi-hero-title { font-size: 1.55rem; font-weight: 700; margin: 0; }
.lsi-hero-sub   { font-size: 0.92rem; opacity: 0.72; margin-top: 4px; }

.lsi-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px;
    font-weight: 600; font-size: 0.95rem;
    border: 1px solid rgba(127,127,127,0.28);
}
.lsi-delta { font-size: 0.84rem; opacity: 0.7; margin-left: 8px; }

.lsi-card {
    border: 1px solid rgba(127,127,127,0.22);
    border-radius: 12px;
    padding: 14px 16px;
    background: rgba(127,127,127,0.04);
}
.lsi-section-title {
    font-size: 1.05rem; font-weight: 700; margin: 8px 0 6px;
    letter-spacing: -0.005em;
}
.lsi-muted { opacity: 0.72; font-size: 0.85rem; }
.lsi-strip {
    border-left: 4px solid #888; padding: 6px 12px;
    margin: 6px 0; border-radius: 4px;
    background: rgba(127,127,127,0.05);
}
.lsi-toolbar {
    display: flex; justify-content: flex-end; align-items: center;
    gap: 8px; margin-bottom: 10px; min-height: 44px;
}
.lsi-llm-mod {
    margin-top: 10px; padding: 10px 12px;
    border-radius: 8px;
    background: rgba(31,78,121,0.09);
    border: 1px solid rgba(31,78,121,0.22);
    font-size: 13px; line-height: 1.45;
}
details.lsi-term {
    display: inline-block;
    vertical-align: baseline;
    max-width: 100%;
}
details.lsi-term > summary {
    cursor: pointer;
    list-style: none;
    text-decoration: underline dotted;
    text-underline-offset: 3px;
    text-decoration-color: rgba(90,108,125,0.55);
    color: inherit;
    font: inherit;
    display: inline;
}
details.lsi-term > summary::-webkit-details-marker { display: none; }
details.lsi-term > summary::marker { content: ""; }
details.lsi-term[open] > summary {
    font-weight: 600;
    text-decoration-color: rgba(31,78,121,0.45);
}
details.lsi-term .lsi-term-body {
    display: block;
    margin-top: 6px;
    margin-bottom: 4px;
    padding: 8px 10px;
    font-size: 12px;
    line-height: 1.45;
    font-weight: 400;
    background: rgba(31,78,121,0.07);
    border-radius: 8px;
    border-left: 3px solid rgba(31,78,121,0.35);
    max-width: min(100%, 28rem);
    text-align: left;
}
</style>
""",
    unsafe_allow_html=True,
)


def _get_api_key() -> str:
    return (
        st.session_state.get("deepseek_key")
        or DEEPSEEK_API_KEY_DEFAULT
        or os.getenv("DEEPSEEK_API_KEY", "")
    )


def _llm_chat(prompt: str, system: str, json_mode: bool = True) -> str | None:
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        kwargs = dict(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        rsp = client.chat.completions.create(**kwargs)
        return rsp.choices[0].message.content
    except Exception as e:
        st.error(f"LLM ошибка: {e}")
        return None


DRIVERS = [
    {"module": "m1", "score": "m1_shift_mad", "value": "m1_shift",
     "name": "Спред усреднения резервов", "units": "млрд руб.",
     "up": "Фактические остатки сильно превышают обязательные — банки «копят» ликвидность под стресс.",
     "down": "Фактические остатки близки к обязательным — буфера ликвидности нет."},
    {"module": "m1", "score": "m1_ruo_mad", "value": "m1_ruo",
     "name": "RUONIA", "units": "% годовых",
     "up": "RUONIA выше нормы — стресс межбанка, дорогая овернайт-ликвидность.",
     "down": "RUONIA ниже нормы — избыток ликвидности на межбанке."},
    {"module": "m2", "score": "m2_MAD_score_cover", "value": "m2_Cover_ratio",
     "name": "Cover ratio репо ЦБ", "units": "×",
     "up": "Переспрос на репо ЦБ — банки активно занимают у регулятора, дефицит фондирования.",
     "down": "Низкий спрос на репо — фондирования у банков достаточно."},
    {"module": "m2", "score": "m2_MAD_score_rate_spread", "value": "m2_Rate_spread",
     "name": "Спред ставки репо к ключевой", "units": "п.п.",
     "up": "Ставка отсечения близка к верхней границе коридора — банки готовы платить дороже.",
     "down": "Спред около нуля — стресса нет."},
    {"module": "m3", "score": "m3_mad_score_cover", "value": "m3_cover_ratio",
     "name": "Cover ratio ОФЗ", "units": "×",
     "up": "Переспрос на ОФЗ — у банков избыток ликвидности, идут в safe-haven.",
     "down": "Недоспрос на ОФЗ — банки не готовы вкладываться в длинный долг, локальный дефицит."},
    {"module": "m4", "score": "_m4_tax_kick", "value": "m4_tax_event_weight",
     "name": "Налоговая нагрузка недели", "units": "",
     "up": "Идёт «тяжёлая» налоговая неделя — отток средств клиентов на уплату налогов.",
     "down": "Налогового давления нет."},
    {"module": "m5", "score": "m5_MAD_score_liquidity_deficit", "value": "m5_liquidity_deficit",
     "name": "Структурный дефицит ликвидности (ЦБ)", "units": "млрд руб.",
     "up": "Дефицит структурной ликвидности — банки в долгу перед ЦБ.",
     "down": "Профицит структурной ликвидности — банки размещают излишки."},
    {"module": "m5", "score": "m5_MAD_score_treasury_pressure", "value": "m5_treasury_pressure",
     "name": "Давление казначейства (изменение за день)", "units": "млрд руб.",
     "up": "Резкий отток средств казначейства — деньги уходят с корсчетов банков.",
     "down": "Приток средств казначейства — ликвидность возвращается в систему."},
]


def _build_llm_context_blocks(
    *,
    cur_dt: pd.Timestamp,
    lsi_value: float,
    status_emoji: str,
    status_label: str,
    delta_7: float,
    delta_30: float,
    panel_pos: int,
    wide_pos: int,
    lsi_series: pd.Series,
    wide: pd.DataFrame,
    wide_at_date: pd.Series,
    wide_dates: pd.Series,
    events_df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Собирает текстовый контекст для LLM: строки ctx_lines + списки событий/налогов.

    Используется и для обзора дня, и для пояснений по модулям (один и тот же набор фактов).
    """
    def _build_module_lines(module_prefix: str) -> list[str]:
        lines: list[str] = []
        for d in DRIVERS:
            if d["module"] != module_prefix:
                continue
            if d["score"] not in wide.columns:
                continue
            sc = wide_at_date.get(d["score"])
            if pd.isna(sc):
                continue
            stats = _series_stats(wide, wide_pos, d.get("value"), HIST_WINDOW)
            if not _is_alive_signal(float(sc), stats):
                continue
            parts = [
                f"{d['name']}: оценка необычности (0 = как обычно, чем дальше, тем реже) = {float(sc):+.2f}"
            ]
            for k in ("current", "median", "delta_7d", "delta_30d"):
                label = STAT_LABELS_FOR_LLM.get(k, k)
                v = stats.get(k)
                if v is not None and pd.notna(v):
                    parts.append(f"{label}={float(v):.4g} {d['units']}".rstrip())
            pct = stats.get("percentile")
            if pct is not None and pd.notna(pct):
                parts.append(
                    f"место среди прошлого года={int(round(pct * 100))} из 100 "
                    f"(сколько процентов дней за год показатель был не выше сегодняшнего)"
                )
            lines.append("- " + "; ".join(parts))
        return lines

    prior_events: list[str] = []
    upcoming_events: list[str] = []
    if not events_df.empty:
        ev_sorted = events_df.copy()
        ev_sorted["delta_days"] = (pd.to_datetime(ev_sorted["date"]) - cur_dt).dt.days
        prior_evs = ev_sorted[(ev_sorted["delta_days"] <= 0) & (ev_sorted["delta_days"] >= -180)]
        for _, r in prior_evs.sort_values("delta_days", ascending=False).iterrows():
            prior_events.append(
                f"{pd.Timestamp(r['date']).strftime('%Y-%m-%d')} "
                f"({int(r['delta_days']):+d}д до выбранной даты): {r['label']}"
            )
        next_evs = ev_sorted[(ev_sorted["delta_days"] > 0) & (ev_sorted["delta_days"] <= 60)]
        for _, r in next_evs.sort_values("delta_days").iterrows():
            upcoming_events.append(
                f"{pd.Timestamp(r['date']).strftime('%Y-%m-%d')} "
                f"(через {int(r['delta_days'])}д): {r['label']}"
            )

    look_back = min(30, panel_pos)
    if look_back > 5:
        slope = float(
            (lsi_series.iloc[panel_pos] - lsi_series.iloc[panel_pos - look_back]) / look_back
        )
        trend_word = "растёт" if slope > 0.05 else ("снижается" if slope < -0.05 else "стабилен")
        trend_line = (
            f"Тренд LSI: {trend_word} (наклон {slope:+.2f}/день за последние {look_back} дней)."
        )
    else:
        trend_line = "Тренд LSI: данных мало для оценки."

    upcoming_tax: list[str] = []
    if "m4_tax_event_weight" in wide.columns:
        future_mask = (
            (wide_dates > cur_dt) & (wide_dates <= cur_dt + pd.Timedelta(days=14))
        )
        for _, r in wide.loc[future_mask].iterrows():
            w = float(r.get("m4_tax_event_weight") or 0.0)
            if w <= 0.2:
                continue
            tax_cols = [
                c.replace("m4_", "")
                for c in wide.columns
                if c.startswith("m4_") and c not in {
                    "m4_year", "m4_month", "m4_day", "m4_n_events_html",
                    "m4_n_important", "m4_tax_event_weight", "m4_day_type_code",
                }
                and pd.notna(r.get(c)) and float(r.get(c) or 0) > 0
            ]
            taxes = ", ".join(tax_cols[:4]) if tax_cols else "налоговый день"
            upcoming_tax.append(
                f"{pd.Timestamp(r['date']).strftime('%Y-%m-%d')} (вес {w:.2f}): {taxes}"
            )
        upcoming_tax = upcoming_tax[:5]

    ctx_lines = [
        f"Дата разбора: {cur_dt.strftime('%Y-%m-%d')}",
        f"LSI: {lsi_value:.1f}/100, статус: {status_emoji} {status_label}",
        f"Изменение LSI за 7 дней: {delta_7:+.1f}; за 30 дней: {delta_30:+.1f}",
        trend_line,
        "",
        "Показатели по модулям (оценка необычности и фактические числа на дату):",
    ]
    for code in ("m1", "m2", "m3", "m4", "m5"):
        ctx_lines.append(f"[{code.upper()} · {MOD_LABELS[code]}]")
        lines = _build_module_lines(code)
        ctx_lines.extend(lines if lines else ["- (нет заметных отклонений от обычного уровня)"])

    if prior_events:
        ctx_lines.append("")
        ctx_lines.append("События ДО выбранной даты (могут объяснять текущее состояние):")
        ctx_lines.extend(f"- {x}" for x in prior_events[:10])

    if upcoming_events:
        ctx_lines.append("")
        ctx_lines.append("Предстоящие макрособытия (в ближайшие 60 дней):")
        ctx_lines.extend(f"- {x}" for x in upcoming_events)

    if upcoming_tax:
        ctx_lines.append("")
        ctx_lines.append("Налоговые дни в ближайшие 14 дней:")
        ctx_lines.extend(f"- {x}" for x in upcoming_tax)

    return ctx_lines, prior_events, upcoming_events, upcoming_tax


def _shift(series: pd.Series) -> pd.Series:
    return (series + LSI_SHIFT).clip(0, 100)


def _load_panel() -> pd.DataFrame:
    panel_path = DATA_DIR / "lsi_panel.csv"
    if panel_path.exists():
        out = pd.read_csv(panel_path, parse_dates=["date"])
    else:
        wide_local = load_wide_lsi()
        lsi = lsi_column(wide_local)
        out = pd.DataFrame({"date": wide_local["date"], "lsi_raw": wide_local[lsi].clip(0, 100)})
        out = out.dropna(subset=["lsi_raw"]).reset_index(drop=True)

    out["lsi_smooth"] = (
        out["lsi_raw"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    )
    std = out["lsi_raw"].rolling(21, min_periods=3, center=True).std().fillna(0.0)
    out["lsi_lo"] = (out["lsi_smooth"] - std).clip(0, 100)
    out["lsi_hi"] = (out["lsi_smooth"] + std).clip(0, 100)
    return out


def _fmt(v: float | None, units: str = "", digits: int = 3) -> str:
    if v is None or pd.isna(v):
        return "—"
    fv = float(v)
    if abs(fv) >= 1000:
        s = f"{fv:,.0f}".replace(",", " ")
    elif abs(fv) >= 100:
        s = f"{fv:.1f}"
    elif abs(fv) >= 10:
        s = f"{fv:.2f}"
    else:
        s = f"{fv:.{digits}g}"
    return f"{s} {units}".rstrip()


def _series_stats(wide_df: pd.DataFrame, idx: int, col: str, window: int = HIST_WINDOW) -> dict:
    if col not in wide_df.columns:
        return {}
    s = pd.to_numeric(wide_df[col], errors="coerce")
    cur = s.iloc[idx]
    lo = max(0, idx - window + 1)
    hist = s.iloc[lo : idx + 1].dropna()
    out: dict = {"current": cur}
    if not hist.empty:
        out["median"] = float(hist.median())
        rank = (hist <= cur).sum() / len(hist) if pd.notna(cur) else None
        out["percentile"] = float(rank) if rank is not None else None
    for back, key in [(7, "delta_7d"), (30, "delta_30d")]:
        j = max(0, idx - back)
        prev = s.iloc[j]
        if pd.notna(cur) and pd.notna(prev):
            out[key] = float(cur) - float(prev)
    return out


def _fmt_delta(v: float | None, units: str = "", *, ref: float | None = None) -> str:
    """Δ-значение с явным знаком, теми же правилами что и _fmt.

    Особенности:
      * NaN → «—»;
      * «микро-Δ» (|v| < 1% от референса ref или абсолютно < 1e-3) → «≈ 0»,
        чтобы не отображать «+0 млрд руб.» на ступенчатых рядах (m1_shift
        обновляется раз в месяц — между обновлениями Δ7д = 0 действительно).
    """
    if v is None or pd.isna(v):
        return "—"
    fv = float(v)
    eps = 1e-3
    if ref is not None and pd.notna(ref):
        eps = max(eps, abs(float(ref)) * 0.01)
    if abs(fv) <= eps:
        return f"≈ 0 {units}".rstrip()
    sign = "+" if fv >= 0 else "−"
    av = abs(fv)
    if av >= 1000:
        s = f"{av:,.0f}".replace(",", " ")
    elif av >= 100:
        s = f"{av:.1f}"
    elif av >= 10:
        s = f"{av:.2f}"
    else:
        s = f"{av:.3g}"
    return f"{sign}{s} {units}".rstrip()


def _driver_mini_chart(
    wide_df: pd.DataFrame,
    pos: int,
    val_col: str,
    title: str,
    *,
    direction: int,
    lookback: int = 90,
    lookforward: int = 14,
    as_candles: bool = False,
) -> go.Figure | None:
    """Маленький график значения драйвера вокруг выбранной даты.

    Содержит:
      * линию значения за окно [-lookback, +lookforward];
        либо при ``as_candles=True`` — японские свечи из синтетического OHLC
        (open = предыдущее значение, close = текущее; зелёный вверх, красный вниз);
      * штриховую горизонтальную медиану за это окно;
      * жирный маркер на выбранной дате;
      * закрашенную IQR-полосу (25–75 перцентилей окна), чтобы видеть «норму»;
      * цвет линии — серый, маркер — красный (если фактор сдвигает LSI вверх) или
        зелёный (если вниз), чтобы аналитик глазами видел знак.
    """
    if val_col not in wide_df.columns:
        return None
    lo = max(0, pos - lookback)
    hi = min(len(wide_df), pos + lookforward + 1)
    window = wide_df.iloc[lo:hi]
    y = pd.to_numeric(window[val_col], errors="coerce")
    if y.dropna().empty:
        return None

    marker_color = "#b03a2e" if direction >= 0 else "#1e8a4f"

    fig = go.Figure()

    q25, q75 = float(y.quantile(0.25)), float(y.quantile(0.75))
    if np.isfinite(q25) and np.isfinite(q75) and q25 != q75:
        fig.add_hrect(
            y0=min(q25, q75), y1=max(q25, q75),
            fillcolor="rgba(127,127,127,0.10)", line_width=0, layer="below",
        )
    med = float(y.median())
    if np.isfinite(med):
        fig.add_hline(y=med, line=dict(color="#999", width=1, dash="dot"))

    drew_candles = False
    if as_candles:
        ohlc_df = synthetic_ohlc_from_close(y, window["date"])
        if ohlc_df is not None and not ohlc_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=ohlc_df["dt"],
                    open=ohlc_df["o"],
                    high=ohlc_df["h"],
                    low=ohlc_df["l"],
                    close=ohlc_df["c"],
                    increasing_line_color="#1e8a4f",
                    decreasing_line_color="#b03a2e",
                    increasing_fillcolor="rgba(30, 138, 79, 0.35)",
                    decreasing_fillcolor="rgba(176, 58, 46, 0.35)",
                    whiskerwidth=0.7,
                    name="",
                    hovertemplate="%{x|%Y-%m-%d}<br>O=%{open:.4g} H=%{high:.4g}"
                    "<br>L=%{low:.4g} C=%{close:.4g}<extra></extra>",
                    showlegend=False,
                )
            )
            drew_candles = True

    if not drew_candles:
        fig.add_trace(
            go.Scatter(
                x=window["date"], y=y,
                mode="lines",
                line=dict(color="#5a6c7d", width=1.6),
                connectgaps=True,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4g}<extra></extra>",
                showlegend=False,
            )
        )
    cur_val = pd.to_numeric(wide_df.iloc[pos][val_col], errors="coerce")
    cur_date = wide_df.iloc[pos]["date"]
    if pd.notna(cur_val):
        fig.add_trace(
            go.Scatter(
                x=[cur_date], y=[float(cur_val)],
                mode="markers",
                marker=dict(size=13, color=marker_color, line=dict(color="white", width=2)),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.4g}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_vline(
            x=pd.Timestamp(cur_date),
            line=dict(color=marker_color, width=1, dash="dash"),
            opacity=0.5,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=11), x=0, y=0.97),
        height=170,
        margin=dict(l=8, r=8, t=24, b=22),
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=9), rangeslider=dict(visible=False)),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(127,127,127,0.18)",
            tickfont=dict(size=9), zeroline=False,
        ),
        hovermode="x",
    )
    return fig


def _is_alive_signal(score: float, stats: dict, *, mad_floor: float = 0.15) -> bool:
    """Сигнал считается «живым», если хотя бы одно из:
    1) |MAD| ≥ mad_floor (есть аномалия относительно истории);
    2) есть заметное движение (|Δ7д| или |Δ30д| > 0);
    3) текущее значение явно не нулевое и не равно медиане.

    Это режет «мёртвые» ряды, где данных нет / forward-fill, но MAD = 0.
    """
    if pd.isna(score):
        return False
    if abs(float(score)) >= mad_floor:
        return True
    cur = stats.get("current")
    med = stats.get("median")
    d7 = stats.get("delta_7d")
    d30 = stats.get("delta_30d")
    moved = (d7 is not None and pd.notna(d7) and abs(float(d7)) > 1e-9) or (
        d30 is not None and pd.notna(d30) and abs(float(d30)) > 1e-9
    )
    if moved:
        return True
    if cur is not None and pd.notna(cur) and med is not None and pd.notna(med):
        return abs(float(cur) - float(med)) > 1e-6 and abs(float(cur)) > 1e-9
    return False


panel = _load_panel()
wide = prepare_wide_lsi_for_charts()
events_path = DATA_DIR / "stress_events.csv"
events = (
    pd.read_csv(events_path, parse_dates=["date"])
    if events_path.exists()
    else pd.DataFrame(columns=["date", "label"])
)

latest_panel_pos = len(panel) - 1
latest_lsi = max(0.0, min(100.0, float(panel["lsi_smooth"].iloc[-1]) + LSI_SHIFT))
latest_date = pd.Timestamp(panel["date"].iloc[-1]).strftime("%Y-%m-%d")
latest_label, latest_emoji, _ = status_for(latest_lsi)
latest_color = STATUS_COLORS.get(latest_emoji, "#555")

latest_d7 = float(
    (panel["lsi_smooth"].iloc[-1] - panel["lsi_smooth"].iloc[max(0, latest_panel_pos - 7)])
)
latest_d30 = float(
    (panel["lsi_smooth"].iloc[-1] - panel["lsi_smooth"].iloc[max(0, latest_panel_pos - 30)])
)

_hdr_pad, _hdr_pop = st.columns([3, 1])
with _hdr_pop:
    with st.popover("⚙️ Настройки", width="stretch"):
        st.caption("LLM-объяснение через DeepSeek API")
        st.session_state["deepseek_key"] = st.text_input(
            "API key",
            type="password",
            value=_get_api_key(),
            placeholder="sk-…",
            help="Ключ DeepSeek. Можно задать через env var DEEPSEEK_API_KEY.",
            key="deepseek_key_input",
        )
        st.caption(f"`{DEEPSEEK_MODEL}` · `{DEEPSEEK_BASE_URL}`")
        if _get_api_key():
            st.success("Ключ задан — LLM включён", icon="✅")
        else:
            st.info("Без ключа блоки LLM не сгенерируются.", icon="ℹ️")

st.markdown(
    f"""
<div class="lsi-hero">
  <div style="flex:1; min-width:0">
    <div class="lsi-hero-title">Liquidity Stress Index</div>
    <div class="lsi-hero-sub">Система раннего предупреждения стресса рублёвой ликвидности · ПСБ</div>
  </div>
  <div style="text-align:right; flex-shrink:0">
    <div class="lsi-pill" style="border-color:{latest_color}; color:{latest_color}">
      {latest_emoji} {latest_label} · <b>{latest_lsi:0.1f}</b>/100
    </div>
    <div class="lsi-delta">на {latest_date} · {term_abbr("lsi_change_7d", label="7 дн.")}: {latest_d7:+.1f} · {term_abbr("lsi_change_30d", label="30 дн.")}: {latest_d30:+.1f}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="lsi-section-title">'
    "Глобальный LSI · «оценка относительно всей доступной истории рынка»"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="lsi-muted">Итоговый показатель модели <code>LSI_lgbm_tuned</code>: насколько '
    "напряжена ликвидность сейчас по сравнению со <b>всей накопленной историей</b> (шкала 0–100). "
    "Клик по точке на графике — ниже открывается разбор факторов на выбранную дату.</div>",
    unsafe_allow_html=True,
)

lsi_raw = _shift(panel["lsi_raw"])
lsi_smooth = _shift(panel["lsi_smooth"])

fig_lsi = go.Figure()
fig_lsi.update_layout(shapes=status_band_shapes())

fig_lsi.add_trace(
    go.Scatter(
        x=panel["date"], y=lsi_raw,
        mode="lines", name="LSI без сглаживания",
        line=dict(color="#6c757d", width=1),
        hovertemplate="%{x|%Y-%m-%d}<br>LSI без сглаживания=%{y:.1f}<extra></extra>",
    )
)
fig_lsi.add_trace(
    go.Scatter(
        x=panel["date"], y=lsi_smooth,
        mode="lines", name=f"LSI сглаженный ({SMOOTH_WINDOW} дн.)",
        line=dict(color="#0a3d62", width=2.4),
        hovertemplate="%{x|%Y-%m-%d}<br>LSI сглаженный=%{y:.1f}<extra></extra>",
    )
)

fig_lsi.add_trace(
    go.Scatter(
        x=panel["date"], y=lsi_smooth,
        mode="markers", name="Клик по дате",
        marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=0)),
        hovertemplate="%{x|%Y-%m-%d}<extra>клик для разбора</extra>",
        showlegend=False,
    )
)

for _, row in events.iterrows():
    x = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    fig_lsi.add_shape(
        type="line", x0=x, x1=x, y0=0, y1=100,
        xref="x", yref="y",
        line=dict(color="#b03a2e", width=1, dash="dash"),
    )

if not events.empty:
    ev_dates = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
    fig_lsi.add_trace(
        go.Scatter(
            x=ev_dates, y=[97] * len(events),
            mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="#b03a2e",
                        line=dict(width=1, color="#7a1f17")),
            name="События",
            text=events["label"].astype(str).tolist(),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{text}<extra></extra>",
            showlegend=False,
        )
    )

fig_lsi.update_layout(
    yaxis=dict(range=[0, 100], title="Глобальный LSI, 0–100 (вся доступная история)"),
    xaxis_title="",
    hovermode="closest",
    clickmode="event+select",
    height=540,
    margin=dict(l=10, r=10, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

event_global = st.plotly_chart(
    fig_lsi, width="stretch",
    on_select="rerun",
    selection_mode=("points", "box", "lasso"),
    key="lsi_chart",
)


def _build_local_panel() -> pd.DataFrame | None:
    """Возвращает локальный LSI-панель: date + lsi_raw/smooth/lo/hi.

    Сначала ищем `lsi_panel_local.csv` (быстрый путь), затем фоллбэк —
    `LSI_lgbm_local_multi` из `wide_lsi` (если экспорт ещё не пересчитан).
    """
    p = load_lsi_panel_local()
    if p is not None and not p.empty:
        return p
    col = local_lsi_column(wide)
    if col is None:
        return None
    out = wide[["date", col]].dropna(subset=[col]).copy()
    out = out.rename(columns={col: "lsi_raw"})
    out["lsi_raw"] = out["lsi_raw"].clip(0.0, 100.0)
    out["lsi_smooth"] = (
        out["lsi_raw"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    )
    std = out["lsi_raw"].rolling(21, min_periods=3, center=True).std().fillna(0.0)
    out["lsi_lo"] = (out["lsi_smooth"] - std).clip(0.0, 100.0)
    out["lsi_hi"] = (out["lsi_smooth"] + std).clip(0.0, 100.0)
    return out.reset_index(drop=True)


panel_local = _build_local_panel()


def _build_local_figure(p_local: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(shapes=status_band_shapes())
    fig.add_trace(
        go.Scatter(
            x=p_local["date"], y=p_local["lsi_raw"],
            mode="lines", name="Локальный LSI без сглаживания",
            line=dict(color="#6c757d", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>локальный без сглаживания=%{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p_local["date"], y=p_local["lsi_smooth"],
            mode="lines", name=f"Локальный LSI сглаженный ({SMOOTH_WINDOW} дн.)",
            line=dict(color="#0a6b3d", width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>локальный сглаженный=%{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p_local["date"], y=p_local["lsi_smooth"],
            mode="markers", name="Клик по дате",
            marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=0)),
            hovertemplate="%{x|%Y-%m-%d}<extra>клик для разбора</extra>",
            showlegend=False,
        )
    )
    for _, row in events.iterrows():
        x = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        fig.add_shape(
            type="line", x0=x, x1=x, y0=0, y1=100,
            xref="x", yref="y",
            line=dict(color="#b03a2e", width=1, dash="dash"),
        )
    fig.update_layout(
        yaxis=dict(
            range=[0, 100],
            title="Локальный LSI, 0–100 (текущий рыночный режим, ~1 год)",
        ),
        xaxis_title="",
        hovermode="closest",
        clickmode="event+select",
        height=460,
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


event_local = None
if panel_local is not None and not panel_local.empty:
    st.markdown(
        '<div class="lsi-section-title" style="margin-top:18px">'
        "Локальный LSI · «оценка относительно текущего рыночного режима»"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="lsi-muted">Три модели с разной «памятью» (около четверти года, полгода и год), '
        "затем усреднение — в данных это <code>LSI_lgbm_local_multi</code>. "
        "Смысл: **насколько сегодняшнее положение необычно в рамках последнего года** "
        "(скользящее окно ~252 торговых дня), без доминирования очень долгого тренда. "
        "В паре с глобальным графиком: глобальный — масштаб <b>всей истории</b>, локальный — "
        "<b>текущий рыночный режим</b> на горизонте ~12 месяцев. Клик по точке здесь так же "
        "открывает разбор факторов.</div>",
        unsafe_allow_html=True,
    )
    fig_local = _build_local_figure(panel_local)
    event_local = st.plotly_chart(
        fig_local, width="stretch",
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        key="lsi_local_chart",
    )
else:
    st.info(
        "Локальный LSI ещё не выгружен — пересчитайте экспорт-ячейку `final.ipynb` "
        "после блока multi-window (`LSI_lgbm_local_multi`)."
    )


panel_dates = pd.to_datetime(panel["date"]).dt.normalize()
wide_dates = pd.to_datetime(wide["date"]).dt.normalize()


def _extract_selected_date(ev) -> pd.Timestamp | None:
    if ev is None:
        return None

    def _attr_or_key(o, k):
        if o is None:
            return None
        if isinstance(o, dict):
            return o.get(k)
        return getattr(o, k, None)

    sel = _attr_or_key(ev, "selection")
    pts = _attr_or_key(sel, "points") or []
    if not pts:
        return None
    p0 = pts[0]
    x = _attr_or_key(p0, "x")
    if x is None:
        return None
    try:
        return pd.Timestamp(x).normalize()
    except Exception:
        return None


date_global = _extract_selected_date(event_global)
date_local = _extract_selected_date(event_local)

prev_g = st.session_state.get("__lsi_prev_global_date")
prev_l = st.session_state.get("__lsi_prev_local_date")
prev_selected = st.session_state.get("__lsi_selected_date")

new_selected = prev_selected
chart_clicked = st.session_state.get("__lsi_clicked_chart", "—")
if date_local is not None and date_local != prev_l:
    new_selected = date_local
    chart_clicked = "local"
elif date_global is not None and date_global != prev_g:
    new_selected = date_global
    chart_clicked = "global"

if date_global is not None:
    st.session_state["__lsi_prev_global_date"] = date_global
if date_local is not None:
    st.session_state["__lsi_prev_local_date"] = date_local
if new_selected is not None:
    st.session_state["__lsi_selected_date"] = new_selected
    st.session_state["__lsi_clicked_chart"] = chart_clicked

if new_selected is None:
    selected_date = panel_dates.max()
    selection_note = (
        "Сейчас показан разбор за последний доступный день. "
        "Кликни по точке на любом из графиков выше — обновится."
    )
else:
    selected_date = new_selected
    _label = {"global": "глобальный LSI", "local": "локальный LSI"}.get(chart_clicked, "")
    selection_note = (
        f"Дата разбора: **{selected_date.strftime('%Y-%m-%d')}**"
        + (f" · клик по графику «{_label}»" if _label else "")
    )

wide_pos = int((wide_dates - selected_date).abs().argsort()[0])
panel_pos = int((panel_dates - selected_date).abs().argsort()[0])
wide_at_date = wide.iloc[wide_pos]
panel_at_date = panel.iloc[panel_pos]

lsi_value = max(0.0, min(100.0, float(panel_at_date["lsi_smooth"]) + LSI_SHIFT))
status_label, status_emoji, _ = status_for(lsi_value)

lsi_series = panel["lsi_smooth"] + LSI_SHIFT
delta_7 = float(lsi_series.iloc[panel_pos] - lsi_series.iloc[max(0, panel_pos - 7)])
delta_30 = float(lsi_series.iloc[panel_pos] - lsi_series.iloc[max(0, panel_pos - 30)])

st.markdown(
    '<div class="lsi-section-title">Что влияет на LSI в выбранную дату</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="lsi-muted">{selection_note}</div>', unsafe_allow_html=True)

local_lsi_value: float | None = None
local_status_label: str | None = None
local_status_emoji: str | None = None
if panel_local is not None and not panel_local.empty:
    _ld = pd.to_datetime(panel_local["date"]).dt.normalize()
    _li = int((_ld - selected_date).abs().argsort()[0])
    _lv = float(panel_local["lsi_smooth"].iloc[_li])
    local_lsi_value = max(0.0, min(100.0, _lv))
    local_status_label, local_status_emoji, _ = status_for(local_lsi_value)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 2])
c1.metric(
    f"Глобальный LSI · {pd.Timestamp(wide_at_date['date']).strftime('%Y-%m-%d')}",
    f"{lsi_value:0.1f} / 100",
    delta=f"{status_emoji} {status_label}",
    help=metric_help("global_lsi"),
)
if local_lsi_value is not None:
    c2.metric(
        "Локальный индекс (последний год)",
        f"{local_lsi_value:0.1f} / 100",
        delta=f"{local_status_emoji} {local_status_label}",
        help=metric_help("local_lsi"),
    )
else:
    c2.metric("Локальный индекс", "—", delta="нет данных", help=metric_help("local_lsi"))
c3.metric(
    "Изменение глобального LSI за 7 дней",
    f"{delta_7:+.1f}",
    help=metric_help("lsi_change_7d"),
)
c3.metric(
    "Изменение глобального LSI за 30 дней",
    f"{delta_30:+.1f}",
    help=metric_help("lsi_change_30d"),
)
c4.caption(
    "**Глобальный LSI** — оценка напряжённости ликвидности **относительно всей доступной истории рынка** (0–100). "
    "**Локальный LSI** — оценка **относительно текущего рыночного режима** на скользящем окне ~1 год "
    "(те же 0–100, но слабее тянет за собой многолетний тренд). "
    "Если глобальный в жёлтой зоне, а локальный в зелёной, в долгом горизонте напряжение выше, чем «на последнем году». "
    "Ниже — обзор дня (LLM) и факторы по модулям M1–M5."
)

with st.popover("Расшифровка метрик над графиком (нажмите здесь)", width="stretch"):
    st.caption(
        "У метрик с иконкой ⓘ подсказка часто только при **наведении** на иконку. "
        "Здесь тот же текст — открывается по **клику**."
    )
    st.markdown("##### Глобальный LSI")
    st.markdown(metric_help("global_lsi"))
    st.markdown("##### Локальный индекс")
    st.markdown(metric_help("local_lsi"))
    st.markdown("##### Изменение за 7 дней")
    st.markdown(metric_help("lsi_change_7d"))
    st.markdown("##### Изменение за 30 дней")
    st.markdown(metric_help("lsi_change_30d"))

cur_dt = pd.Timestamp(wide_at_date["date"])
cache_key = cur_dt.strftime("%Y-%m-%d")
ctx_lines, prior_events, upcoming_events, upcoming_tax = _build_llm_context_blocks(
    cur_dt=cur_dt,
    lsi_value=lsi_value,
    status_emoji=status_emoji,
    status_label=status_label,
    delta_7=delta_7,
    delta_30=delta_30,
    panel_pos=panel_pos,
    wide_pos=wide_pos,
    lsi_series=lsi_series,
    wide=wide,
    wide_at_date=wide_at_date,
    wide_dates=wide_dates,
    events_df=events,
)
context_text = "\n".join(ctx_lines)

st.markdown(
    '<div class="lsi-section-title">Обзор дня · LLM</div>',
    unsafe_allow_html=True,
)
_api_llm = bool(_get_api_key())
b_day, cap_day = st.columns([1, 4])
with b_day:
    gen_day = st.button("Сгенерировать обзор дня", key="llm_gen_day_overview", width="stretch")
with cap_day:
    if not _api_llm:
        st.caption("Укажите API-ключ в «⚙️ Настройки» сверху справа.")
    else:
        st.caption(
            f"Контекст: {len(ctx_lines)} строк · событий до даты: {len(prior_events)} · "
            f"впереди: {len(upcoming_events)} · налоговых окон: {len(upcoming_tax)}"
        )

LLM_SYSTEM_BASE = (
    "Ты — старший аналитик казначейства российского системно значимого банка (ПСБ). "
    "Объясняешь состояние рублёвой ликвидности только по цифрам и событиям из контекста. "
    "Запрещено выдумывать данные. Тон: краткий, профессиональный, по-русски. "
    "Во всех текстах для человека избегай матстатистического жаргона (MAD, перцентиль, IQR, σ, корреляция) "
    "без перевода: используй формулировки из блока ЛЕКСИКОН во входном сообщении. "
    "Технические имена колонок не цитируй."
)

render_glossary_llm_expander(api_ok=_api_llm, llm_chat=_llm_chat, system_prompt=LLM_SYSTEM_BASE)

day_cache: dict = st.session_state.setdefault("llm_day_overview", {})
legacy_full: dict = st.session_state.get("llm_cache") or {}
if cache_key in legacy_full and cache_key not in day_cache:
    _leg = legacy_full[cache_key]
    if isinstance(_leg, dict) and any(k in _leg for k in ("general", "reasons", "news_context", "outlook")):
        day_cache[cache_key] = {
            "general": _leg.get("general", ""),
            "reasons": _leg.get("reasons", ""),
            "news_context": _leg.get("news_context", ""),
            "outlook": _leg.get("outlook", ""),
        }

USER_DAY = (
    "Ниже — фактические данные системы на выбранную дату (LSI, модули M1–M5, события, налоги).\n\n"
    + context_text
    + "\n\n"
    + llm_lexicon_block()
    + "\n\nВерни СТРОГО валидный JSON только с ключами general, reasons, news_context, outlook:\n"
    + "{\n"
    + '  "general": "1–2 фразы: что вцелом с ликвидностью в этот день",\n'
    + '  "reasons": "2–4 предложения: почему LSI на этом уровне — только с числами из контекста",\n'
    + '  "news_context": "1–3 предложения: релевантные события из списка «ДО даты»; иначе скажи что рядом нет",\n'
    + '  "outlook": "1–3 предложения: ближайшие 1–4 недели по налогам и событиям из контекста"\n'
    + "}\n"
)

if gen_day and _api_llm:
    with st.spinner("DeepSeek: обзор дня…"):
        raw_d = _llm_chat(USER_DAY, LLM_SYSTEM_BASE, json_mode=True)
    if raw_d:
        try:
            day_cache[cache_key] = json.loads(raw_d)
        except Exception:
            day_cache[cache_key] = {"general": raw_d, "reasons": "", "news_context": "", "outlook": ""}

day_parsed = day_cache.get(cache_key)
if day_parsed:
    _g = day_parsed.get("general") or ""
    _r = day_parsed.get("reasons") or ""
    _n = day_parsed.get("news_context") or ""
    _o = day_parsed.get("outlook") or ""
    if _g:
        st.markdown(f"**Ситуация:** {_g}")
    if _r:
        st.markdown(f"**Почему такой LSI:** {_r}")
    if _n:
        st.markdown(
            f"<div class='lsi-strip' style='border-left-color:#b03a2e'>"
            f"<b>Новостной контекст</b><br>"
            f"<span style='font-size:13px; opacity:.9'>{html.escape(_n)}</span></div>",
            unsafe_allow_html=True,
        )
    if _o:
        st.markdown(
            f"<div class='lsi-strip' style='border-left-color:#1f4e79'>"
            f"<b>Ожидания</b><br>"
            f"<span style='font-size:13px; opacity:.9'>{html.escape(_o)}</span></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="lsi-section-title" style="margin-top:12px">Влияние модулей M1–M5</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="lsi-muted">Числа и графики — из данных системы. Пояснения LLM по каждому модулю '
    'встраиваются внутрь карточки модуля; сгенерировать их можно кнопкой в начале секции.</div>',
    unsafe_allow_html=True,
)

mod_blurbs: dict = st.session_state.setdefault("llm_module_blurbs", {})
if cache_key in legacy_full and cache_key not in mod_blurbs:
    _legm = legacy_full[cache_key]
    if isinstance(_legm, dict) and _legm.get("modules"):
        mod_blurbs[cache_key] = _legm["modules"]

rows = []
for d in DRIVERS:
    sc_col, val_col = d["score"], d.get("value")
    if sc_col not in wide.columns:
        continue
    sc = wide_at_date.get(sc_col)
    if pd.isna(sc):
        continue
    stats = _series_stats(wide, wide_pos, val_col, HIST_WINDOW) if val_col else {}
    sc_f = float(sc)
    if not _is_alive_signal(sc_f, stats):
        continue
    rows.append(
        {
            "module": d["module"], "name": d["name"], "units": d["units"],
            "value_col": val_col,
            "score": sc_f, "abs": abs(sc_f),
            "explain": d["up"] if sc_f >= 0 else d["down"],
            **stats,
        }
    )

drivers_df = pd.DataFrame(rows).sort_values("abs", ascending=False)

if drivers_df.empty:
    st.info("На эту дату нет заметных отклонений показателей от обычного для них уровня.")
else:
    n_max = len(drivers_df)

    blurbs_row_b, blurbs_row_c = st.columns([1, 4])
    with blurbs_row_b:
        gen_modules = st.button(
            "Сгенерировать пояснения по модулям",
            key="llm_gen_module_blurbs",
            width="stretch",
        )
    with blurbs_row_c:
        if not _api_llm:
            st.caption("Пояснения по модулям появятся после ввода API-ключа.")
        else:
            st.caption(
                "Текст LLM добавляется **внутрь каждой карточки модуля** под драйверами."
            )

    top_n = n_max if n_max <= 3 else st.slider(
        "Сколько топ-сигналов учесть", 3, n_max, min(5, n_max)
    )
    top = drivers_df.head(top_n)

    _c_ch1, _c_ch2 = st.columns([1, 3])
    with _c_ch1:
        show_jp_candles = st.checkbox(
            "Японские свечи",
            key="lsi_jp_candles",
            help=(
                "Свечи из дневного ряда: open — прошлое значение, close — текущее. "
                "Зелёный столбик — выше прошлого, красный — ниже."
            ),
        )
    with _c_ch2:
        st.caption(
            "На мини-графиках справа: при включении показываются **свечи** (красный/зелёный столбик) "
            "вместо линии — по одному значению на день, без внутридневных цен."
        )

    USER_MODULES = (
        "Ниже — те же факты, что в дашборде.\n\n"
        + context_text
        + "\n\n"
        + llm_lexicon_block()
        + "\n\nВерни СТРОГО валидный JSON-объект, где ключи ровно "
        + '"M1","M2","M3","M4","M5" (значения — строки). '
        "В каждой строке 1–3 предложения с конкретными числами из контекста для этого модуля; "
        "пиши простым языком (оценка необычности, типичный уровень, изменение за 7/30 дней, "
        "место среди года), без жаргона MAD/перцентиль/IQR. "
        'Если сигналов нет — буквально «нет заметного отклонения». Никаких других ключей.'
    )

    if gen_modules and _api_llm:
        with st.spinner("DeepSeek: пояснения по модулям…"):
            raw_m = _llm_chat(USER_MODULES, LLM_SYSTEM_BASE, json_mode=True)
        if raw_m:
            try:
                parsed_m = json.loads(raw_m)
                if isinstance(parsed_m, dict):
                    mod_blurbs[cache_key] = parsed_m
            except Exception:
                st.warning("Ответ LLM по модулям не распознан как JSON.")

    mod_for_date = mod_blurbs.get(cache_key) or {}

    module_order: list[str] = []
    grouped: dict[str, list[pd.Series]] = {}
    for _, r in top.iterrows():
        m = r["module"]
        if m not in grouped:
            grouped[m] = []
            module_order.append(m)
        grouped[m].append(r)

    for m in module_order:
        items = grouped[m]
        color = MOD_COLORS.get(m, "#444")
        total_abs = sum(float(x["abs"]) for x in items)
        net_dir = sum(float(x["score"]) for x in items)
        head_arrow = "↑" if net_dir >= 0 else "↓"
        net_word = "в сумме в сторону роста LSI" if net_dir >= 0 else "в сумме в сторону снижения LSI"

        with st.container(border=True):
            head_l, head_r = st.columns([3, 2])
            head_l.markdown(
                f"<div style='border-left:6px solid {color}; padding-left:10px; "
                f"font-size:16px'><b>{module_label_html(m)}</b></div>",
                unsafe_allow_html=True,
            )
            _sum_lbl = term_abbr("bar_strength_sum", label="суммарно")
            head_r.markdown(
                f"<div style='text-align:right; opacity:.78; font-size:13px'>"
                f"{net_word} · {_sum_lbl} <b>{head_arrow} {total_abs:.2f}</b>"
                f" · факторов: {len(items)}</div>",
                unsafe_allow_html=True,
            )

            for i, r in enumerate(items):
                if i > 0:
                    st.markdown(
                        "<hr style='margin:10px 0; border:none; "
                        "border-top:1px solid rgba(127,127,127,0.22)'>",
                        unsafe_allow_html=True,
                    )

                direction = "↑" if r["score"] >= 0 else "↓"
                dir_color = "#b03a2e" if r["score"] >= 0 else "#1e8a4f"
                dir_word = (
                    "выше нормы (сдвигает индекс к большему напряжению)"
                    if r["score"] >= 0
                    else "ниже нормы (ослабляет напряжение по индексу)"
                )
                units = r["units"]
                cur_val = r.get("current")
                cur_str = _fmt(cur_val, units)
                med_str = _fmt(r.get("median"), units)
                d7_str = _fmt_delta(r.get("delta_7d"), units, ref=cur_val)
                d30_str = _fmt_delta(r.get("delta_30d"), units, ref=cur_val)
                pct = r.get("percentile")
                pct_str = (
                    f"<b>{int(round(pct * 100))}</b> из 100 — доля дней за год, когда значение "
                    f"было не выше сегодняшнего"
                    if pct is not None and pd.notna(pct)
                    else "нет оценки «где среди года»"
                )

                left, right = st.columns([1.05, 1.4])

                with left:
                    _anom = term_abbr("anomaly_score", label="необычность")
                    st.markdown(
                        f"<div style='display:flex; align-items:baseline; gap:10px; flex-wrap:wrap'>"
                        f"<span style='font-size:15px; font-weight:700'>"
                        f"<span style='color:{dir_color}'>{direction}</span> {inline_hints(r['name'])}</span>"
                        f"<span style='font-size:12px; opacity:.65'>{_anom}: "
                        f"<b style='color:{dir_color}'>{r['score']:+.2f}</b></span>"
                        f"</div>"
                        f"<div style='font-size:12px; opacity:.68; margin-top:-2px'>"
                        f"{dir_word}</div>",
                        unsafe_allow_html=True,
                    )
                    n1, n2 = st.columns(2)
                    _med_l = term_abbr("typical_level_year", label="типичный уровень (~год)")
                    n1.markdown(
                        f"<div style='font-size:11px; opacity:.65; margin-top:8px'>Сейчас</div>"
                        f"<div style='font-size:18px; font-weight:700; color:{dir_color}'>{cur_str}</div>"
                        f"<div style='font-size:11px; opacity:.65; margin-top:6px'>{_med_l}</div>"
                        f"<div style='font-size:14px; font-weight:600'>{med_str}</div>",
                        unsafe_allow_html=True,
                    )
                    _d7 = term_abbr("change_7d", label="за 7 дней")
                    _d30 = term_abbr("change_30d", label="за 30 дней")
                    n2.markdown(
                        f"<div style='font-size:11px; opacity:.65; margin-top:8px'>{_d7}</div>"
                        f"<div style='font-size:14px; font-weight:600'>{d7_str}</div>"
                        f"<div style='font-size:11px; opacity:.65; margin-top:6px'>{_d30}</div>"
                        f"<div style='font-size:14px; font-weight:600'>{d30_str}</div>",
                        unsafe_allow_html=True,
                    )
                    _plc = term_abbr("place_in_year", label="где среди года")
                    st.markdown(
                        f"<div style='font-size:12px; opacity:.78; margin-top:6px'>"
                        f"📊 {_plc}: {pct_str}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='font-size:13px; margin-top:8px; padding:6px 10px;"
                        f" border-left:3px solid {color};"
                        f" background:rgba(127,127,127,0.06); border-radius:4px;'>"
                        f"{inline_hints(str(r['explain']))}</div>",
                        unsafe_allow_html=True,
                    )

                with right:
                    fig_mini = _driver_mini_chart(
                        wide, wide_pos, r["value_col"],
                        title=f"{r['name']} · окно ±90 дней (точка — выбранная дата)",
                        direction=1 if r["score"] >= 0 else -1,
                        as_candles=show_jp_candles,
                    )
                    if fig_mini is not None:
                        st.plotly_chart(
                            fig_mini, width="stretch",
                            config={"displayModeBar": False},
                            key=f"mini_{m}_{i}_{r['value_col']}",
                        )
                    else:
                        st.caption("Нет ряда для графика.")

            llm_mod_txt = mod_for_date.get(m.upper()) or mod_for_date.get(m)
            if llm_mod_txt:
                st.markdown(
                    '<div class="lsi-llm-mod"><b>🤖 Комментарий LLM</b><br>'
                    f"<span>{html.escape(str(llm_mod_txt))}</span></div>",
                    unsafe_allow_html=True,
                )

    by_mod_top = (
        top.groupby("module")["abs"].sum().reindex(MOD_LABELS.keys()).fillna(0.0)
    )
    bar = go.Figure(
        go.Bar(
            x=[MOD_LABELS[m] for m in by_mod_top.index],
            y=by_mod_top.values,
            marker_color=[MOD_COLORS[m] for m in by_mod_top.index],
            text=[f"{v:.1f}" for v in by_mod_top.values],
            textposition="outside",
        )
    )
    bar.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=20, b=20),
        yaxis_title="Сумма оценок необычности (топ-факторы дня)",
        xaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(bar, width="stretch")

st.divider()

metrics_path = DATA_DIR / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    if metrics:
        cols = st.columns(len(metrics))
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, v)

with st.expander("Как читать графики и числа"):
    st.markdown(
        f"""
**Графики LSI**

- **Глобальный LSI** (`LSI_lgbm_tuned`) — **оценка относительно всей доступной истории рынка**, шкала 0–100.
  Если линия годами ползёт вверх, в среднем рынок ощущает большее напряжение ликвидности, чем в прошлом.
  *Синяя* — сглаженный индекс ({SMOOTH_WINDOW} дн., сдвиг +{int(LSI_SHIFT)} для удобной шкалы),
  *серая* — «сырой» прогноз на каждый день без сглаживания.
- **Локальный LSI** (`LSI_lgbm_local_multi`) — **оценка относительно текущего рыночного режима** на скользящем
  окне ~1 год (три модели с памятью ~90 / 180 / 365 дней, затем усреднение). Слабее тянет за собой многолетний тренд;
  удобно отвечать «сейчас относительно спокойно или нет» на горизонте последних месяцев.
- Полоса разброса прогноза (±1 ст. откл. за 21 день) на графиках LSI **отключена** — на графике только
  сглаженная и «сырая» линии. Смысл полосы см. в глоссарии: **нажмите** на подчёркнутые термины или
  «Справка по терминам и вопрос к LLM».
- Зоны статуса: 🟢 `0–40` спокойнее · 🟡 `40–70` повышенное напряжение · 🔴 `≥ 70` остро.
- Красные пунктиры/треугольники — отмеченные стресс-события (подсказка при наведении).

**Драйверы (блок «Что влияет на LSI»)**

- Для каждого фактора M1–M5: значение сейчас, **типичный уровень за год**, изменение за 7 и 30 дней,
  **где значение среди прошлого года** (0–100 из 100), мини-график **±90 дней** вокруг выбранной даты.
- На мини-графике: линия — сам показатель; пунктир — типичный уровень в этом окне; светлая полоса —
  **типичный коридор** (середина 50% дней в окне — от 25-го до 75-го «места»); большая точка — в выбранный день;
  🔴 фактор сдвигает индекс в сторону роста, 🟢 — в сторону снижения напряжённости.
- Число **«необычность»** рядом с названием: насколько сегодняшнее значение выбивается из обычного хода
  (внутри модели — робастная z-оценка по длинному окну). Порядка **1,5** и выше — уже заметно, **3** и выше — очень сильно.
""",
        unsafe_allow_html=True,
    )

st.divider()

contrib_cols = [c for c in panel.columns if c.startswith("contrib_")]
if contrib_cols:
    st.markdown(
        '<div class="lsi-section-title">Вклады модулей в LSI во времени</div>',
        unsafe_allow_html=True,
    )

    panel_stack = panel.copy()
    for c in contrib_cols:
        v = pd.to_numeric(panel_stack[c], errors="coerce").to_numpy()
        is_zero = (v == 0) | np.isnan(v)
        nbr = np.zeros_like(is_zero, dtype=bool)
        if len(v) > 1:
            nbr[1:] |= ~is_zero[:-1]
            nbr[:-1] |= ~is_zero[1:]
        hole = is_zero & nbr
        if hole.any():
            s = pd.Series(v, index=panel_stack.index, dtype=float)
            s[hole] = np.nan
            panel_stack[c] = s.ffill().bfill().fillna(0.0)

    fig_mod = go.Figure()
    for col in contrib_cols:
        key = col.replace("contrib_", "")
        _nm = MOD_LABELS.get(key, key)
        fig_mod.add_trace(
            go.Scatter(
                x=panel_stack["date"], y=panel_stack[col],
                mode="lines",
                name=_nm,
                stackgroup="contrib",
                line=dict(width=0.5, color=MOD_COLORS.get(key, "#888888")),
                fillcolor=MOD_COLORS.get(key, "#888888"),
                connectgaps=True,
                hovertemplate="%{x|%Y-%m-%d}<br>"
                + html.escape(str(_nm))
                + "=%{y:.2f}<extra></extra>",
            )
        )
    fig_mod.update_layout(
        yaxis=dict(title="Вклад модулей в индекс (накопленно по слоям)"),
        xaxis_title="",
        hovermode="x unified",
        height=420,
        margin=dict(l=10, r=10, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_mod, width="stretch")

    st.caption(
        "Накопленная диаграмма: какой модуль в каждый день сильнее всего «тянет» финальный индекс. "
        "Сумма слоёв похожа на сглаженную линию LSI. "
        "В праздники и выходные, когда в сырых данных нули, слои временно переносятся со вчера — "
        "иначе стопка уходила бы в ноль, хотя сам индекс на эти дни посчитан."
    )
