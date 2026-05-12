from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from glossary import TERMS, llm_lexicon_block
from utils import (
    LSI_SMOOTH_WINDOW,
    aligned_shifted_lsi_series_for_wide,
    prepare_wide_lsi_for_charts,
    status_band_shapes,
    synthetic_ohlc_from_close,
)

st.set_page_config(page_title="Модули M1–M5", page_icon="📊", layout="wide")
st.title("Модули M1–M5")
st.caption(
    "Вкладки M1–M5: обзорные графики и корреляции с LSI. Текст про смысл модуля — только от LLM (блок ниже)."
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY_DEFAULT = ""


def _get_api_key() -> str:
    return (
        st.session_state.get("deepseek_key")
        or DEEPSEEK_API_KEY_DEFAULT
        or os.getenv("DEEPSEEK_API_KEY", "")
    )


def _llm_chat(prompt: str, system: str, *, json_mode: bool = True) -> str | None:
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


LLM_SYSTEM_MODULES_PAGE = (
    "Ты — старший аналитик казначейства ПСБ. Пишешь короткие вводные тексты для страницы модулей LSI. "
    "Только правда из контекста; не выдумывай цифры и события. Тон: профессиональный, по-русски. "
    "Избегай сухого жаргона без перевода — ориентир формулировок в блоке ЛЕКСИКОН."
)

MOD_COLORS = {
    "M1": "#1f4e79",
    "M2": "#0a6b3d",
    "M3": "#8e44ad",
    "M4": "#d35400",
    "M5": "#7f6b2c",
}


@dataclass(frozen=True)
class ModuleTab:
    key: str
    short: str
    prefix: str
    primary: tuple[tuple[str, str], ...]
    mad: tuple[tuple[str, str], ...]
    m4_step: bool = False


MODULE_TABS: tuple[ModuleTab, ...] = (
    ModuleTab(
        "M1",
        "Резервы · RUONIA",
        "m1_",
        (
            ("m1_shift", "Запас резервов (факт − норма), млрд ₽"),
            ("m1_ruo", "RUONIA, % годовых"),
        ),
        (("m1_shift_mad", "Необычность: запас резервов"), ("m1_ruo_mad", "Необычность: RUONIA")),
    ),
    ModuleTab(
        "M2",
        "Репо ЦБ",
        "m2_",
        (
            ("m2_Cover_ratio", "Cover ratio репо ЦБ, ×"),
            ("m2_Rate_spread", "Спред ставки репо к ключевой, п.п."),
        ),
        (
            ("m2_MAD_score_cover", "Необычность: cover ratio"),
            ("m2_MAD_score_rate_spread", "Необычность: спред к ключевой"),
        ),
    ),
    ModuleTab(
        "M3",
        "ОФЗ",
        "m3_",
        (("m3_cover_ratio", "Cover ratio ОФЗ, ×"),),
        (("m3_mad_score_cover", "Необычность: cover ОФЗ"),),
    ),
    ModuleTab(
        "M4",
        "Налоги",
        "m4_",
        (("m4_tax_event_weight", "Вес налоговой недели"), ("_m4_tax_kick", "Налоговый сигнал (kick)")),
        (),
        m4_step=True,
    ),
    ModuleTab(
        "M5",
        "Казначейство · ЦБ",
        "m5_",
        (
            ("m5_liquidity_deficit", "Структурный дефицит/профицит, млрд ₽"),
            ("m5_treasury_pressure", "Давление казначейства (Δ), млрд ₽"),
        ),
        (
            ("m5_MAD_score_liquidity_deficit", "Необычность: дефицит"),
            ("m5_MAD_score_treasury_pressure", "Необычность: казначейство"),
        ),
    ),
)


def _default_mod_desc() -> dict[str, str]:
    return {f"M{i}": "" for i in range(1, 6)}


def _module_page_llm_context(wide_win: pd.DataFrame, lsi_win: pd.Series) -> str:
    last = wide_win.iloc[-1]
    d = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")
    lsi_v = float(lsi_win.iloc[-1])
    lines = [
        f"Последняя дата в окне: {d}",
        f"LSI на графиках страницы (как на главной: из панели — rolling({LSI_SMOOTH_WINDOW}) по lsi_raw, затем +20 к шкале): {lsi_v:.2f} из 100.",
        "",
        "По модулям — колонки на вкладке и последнее значение (если есть):",
    ]
    for mod in MODULE_TABS:
        lines.append(f"{mod.key} — вкладка «{mod.short}», префикс колонок `{mod.prefix}`:")
        cols_seen = False
        for c, title in mod.primary + mod.mad:
            if c not in wide_win.columns:
                continue
            cols_seen = True
            v = last.get(c)
            if pd.notna(v):
                lines.append(f"  - {c} («{title}») = {float(v):.4g}")
            else:
                lines.append(f"  - {c} («{title}») = NaN")
        if mod.m4_step:
            lines.append(
                "  (M4: ступенчатые ряды; нули между налоговыми датами — часть модели, не «дыра» в данных.)"
            )
        if not cols_seen:
            lines.append("  (в выгрузке нет колонок этого модуля)")
    return "\n".join(lines)


def _top_features(wide: pd.DataFrame, lsi_aligned: pd.Series, prefix: str, k: int) -> list[tuple[str, float]]:
    feats = [
        c
        for c in wide.columns
        if c.startswith(prefix) and pd.api.types.is_numeric_dtype(wide[c])
    ]
    rows: list[tuple[str, float]] = []
    for c in feats:
        df = pd.DataFrame({"lsi": lsi_aligned.values, "x": wide[c].values}).dropna()
        if len(df) < 50:
            continue
        r = df["lsi"].corr(df["x"])
        if pd.isna(r):
            continue
        rows.append((c, float(r)))
    rows.sort(key=lambda kv: -abs(kv[1]))
    return rows[:k]


def _plot_feature_vs_lsi(
    date: pd.Series,
    lsi_y: pd.Series,
    feat_y: pd.Series,
    feat_name: str,
    corr: float,
    *,
    feat_as_candles: bool = False,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(shapes=status_band_shapes())
    fig.add_trace(
        go.Scatter(
            x=date,
            y=lsi_y,
            name="Глобальный LSI (сглаженный)",
            line=dict(color="#0a3d62", width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>LSI=%{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fs = pd.Series(pd.to_numeric(feat_y, errors="coerce"))
    drew_candles = False
    if feat_as_candles:
        ohlc = synthetic_ohlc_from_close(fs, pd.Series(date))
        if ohlc is not None and not ohlc.empty:
            fig.add_trace(
                go.Candlestick(
                    x=ohlc["dt"],
                    open=ohlc["o"],
                    high=ohlc["h"],
                    low=ohlc["l"],
                    close=ohlc["c"],
                    name=feat_name,
                    increasing_line_color="#1e8a4f",
                    decreasing_line_color="#b03a2e",
                    increasing_fillcolor="rgba(30, 138, 79, 0.35)",
                    decreasing_fillcolor="rgba(176, 58, 46, 0.35)",
                    whiskerwidth=0.7,
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>"
                        + feat_name
                        + "<br>O=%{open:.4g} H=%{high:.4g}<br>L=%{low:.4g} C=%{close:.4g}<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )
            drew_candles = True
    if not drew_candles:
        fig.add_trace(
            go.Scatter(
                x=date,
                y=feat_y,
                name=feat_name,
                line=dict(color="#d35400", width=1.6),
                connectgaps=True,
                hovertemplate="%{x|%Y-%m-%d}<br>" + feat_name + "=%{y:.4g}<extra></extra>",
            ),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="LSI, 0–100", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text=feat_name, secondary_y=True, showgrid=False)
    fig.update_layout(
        title=(
            f"{feat_name} · вместе с LSI: {corr:+.2f} "
            f"(−1…+1, «{TERMS['together_with_lsi'].public_name}» — справка на главной)"
        ),
        hovermode="x unified",
        height=340,
        margin=dict(l=10, r=10, t=56, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    return fig


def _module_overview_figure(
    wide: pd.DataFrame,
    lsi: pd.Series,
    mod: ModuleTab,
) -> go.Figure | None:
    prim = [(c, t) for c, t in mod.primary if c in wide.columns]
    mads = [(c, t) for c, t in mod.mad if c in wide.columns]
    if not prim and not mads:
        return None
    rows = 1 + len(prim) + len(mads)
    rh = [0.22] + [0.78 / (rows - 1)] * (rows - 1)
    titles = ("Глобальный LSI (0–100)",) + tuple(t for _, t in prim) + tuple(t for _, t in mads)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=rh,
        subplot_titles=titles,
    )
    fig.update_annotations(font_size=11)
    x = wide["date"]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lsi,
            name="LSI",
            line=dict(color="#0a3d62", width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>LSI=%{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    r = 2
    accent = MOD_COLORS.get(mod.key, "#444")
    accent2 = "#34495e" if mod.key != "M4" else "#a04000"
    for idx, (col, title) in enumerate(prim):
        y = pd.to_numeric(wide[col], errors="coerce")
        line_color = accent if idx == 0 else accent2
        if mod.m4_step:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=title,
                    mode="lines",
                    line=dict(color=line_color, width=2, shape="hv"),
                    connectgaps=False,
                    hovertemplate="%{x|%Y-%m-%d}<br>" + title + "=%{y:.3g}<extra></extra>",
                ),
                row=r,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=title,
                    mode="lines",
                    line=dict(color=line_color, width=2),
                    connectgaps=True,
                    hovertemplate="%{x|%Y-%m-%d}<br>" + title + "=%{y:.4g}<extra></extra>",
                ),
                row=r,
                col=1,
            )
        fig.update_yaxes(title_text="", row=r, col=1)
        r += 1
    for col, title in mads:
        y = pd.to_numeric(wide[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=title,
                mode="lines",
                line=dict(color="#5a6c7d", width=1.5),
                connectgaps=True,
                hovertemplate="%{x|%Y-%m-%d}<br>" + title + "=%{y:+.2f}<extra></extra>",
            ),
            row=r,
            col=1,
        )
        fig.update_yaxes(title_text="", range=[-4, 4], row=r, col=1, zeroline=True, zerolinewidth=1)
        r += 1

    fig.update_layout(
        height=min(280 + rows * 150, 1200),
        margin=dict(l=10, r=10, t=48, b=24),
        showlegend=False,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.12)", row=1, col=1)
    return fig


def _render_module_tab(
    mod: ModuleTab,
    wide: pd.DataFrame,
    lsi_aligned: pd.Series,
    top_k: int,
    use_candles: bool,
    llm_intro: str | None,
) -> None:
    st.markdown(f"### {mod.key}")
    text = (llm_intro or "").strip()
    if text:
        with st.container(border=True):
            st.markdown(text)
    else:
        st.caption("Описание модуля появится после генерации LLM (блок над вкладками).")

    fig_ov = _module_overview_figure(wide, lsi_aligned, mod)
    if fig_ov is not None:
        st.plotly_chart(fig_ov, width="stretch")
    else:
        st.info("Нет колонок для обзорного графика в текущей выгрузке.")

    st.divider()
    st.subheader("Сильнее всего движется вместе с LSI")
    top = _top_features(wide, lsi_aligned, mod.prefix, top_k)
    if not top:
        st.caption("Недостаточно числовых рядов с покрытием для корреляции.")
        return
    st.markdown(
        "**Топ по модулю:** "
        + ", ".join(f"`{c}` (**{r:+.2f}**)" for c, r in top)
    )
    for col, r in top:
        fig = _plot_feature_vs_lsi(
            wide["date"], lsi_aligned, wide[col], col, r, feat_as_candles=use_candles
        )
        st.plotly_chart(fig, width="stretch")


wide = prepare_wide_lsi_for_charts()
lsi_aligned = aligned_shifted_lsi_series_for_wide(wide)

ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.1, 2.2])
with ctrl1:
    last_n = st.slider(
        "Глубина истории (дней)",
        180,
        min(6000, len(wide)),
        min(2200, len(wide)),
        step=60,
        help="Обрезка справа: последние N строк календаря (как на графиках модуля).",
    )
with ctrl2:
    top_k = st.slider("Топ фич по связи с LSI", 1, 4, 2)
with ctrl3:
    use_candles = st.checkbox(
        "Японские свечи на графиках «ряд vs LSI»",
        key="lsi_jp_candles",
        help="Правая ось: свечи из дневного ряда (open = вчера, close = сегодня).",
    )

wide_win = wide.iloc[-last_n:].reset_index(drop=True)
lsi_win = lsi_aligned.iloc[-last_n:].reset_index(drop=True)

_SK_MOD_LLM = "llm_module_page_descriptions"
if _SK_MOD_LLM not in st.session_state:
    st.session_state[_SK_MOD_LLM] = _default_mod_desc()

with st.container(border=True):
    st.markdown("**Описания модулей (LLM)**")
    st.caption(
        f"Ряд LSI на графиках здесь совпадает с главной: из панели — сглаживание `lsi_raw` окном {LSI_SMOOTH_WINDOW} дней, "
        "затем сдвиг шкалы +20; выровнено по датам `wide_lsi`."
    )
    if st.button("Сгенерировать описания модулей (LLM)", type="primary", key="mod_page_llm_gen"):
        if not _get_api_key():
            st.warning(
                "Укажите ключ DeepSeek на главной странице (сайдбар) или переменную окружения DEEPSEEK_API_KEY."
            )
        else:
            ctx = _module_page_llm_context(wide_win, lsi_win)
            user_pt = llm_lexicon_block()
            user_pt += (
                "\n\nЗадача: для каждого ключа M1..M5 один абзац из 3–5 предложений: что модуль измеряет в индексе "
                "ликвидности, зачем он практически аналитику, какие линии на вкладке главные (имена колонок из контекста), "
                "как читать их вместе с LSI. Не придумывай факты и цифры сверх контекста.\n\n"
                f"Контекст выбранного окна (последние {last_n} календарных дней):\n{ctx}\n\n"
                'Верни СТРОГО JSON-объект с ключами "M1","M2","M3","M4","M5" и значениями — строками.'
            )
            raw = _llm_chat(user_pt, LLM_SYSTEM_MODULES_PAGE, json_mode=True)
            if raw:
                try:
                    data = json.loads(raw)
                    out: dict[str, str] = {}
                    for k in ("M1", "M2", "M3", "M4", "M5"):
                        v = data.get(k, "")
                        out[k] = str(v).strip() if v is not None else ""
                    st.session_state[_SK_MOD_LLM] = out
                    st.success("Описания обновлены.")
                except json.JSONDecodeError:
                    st.error("Ответ модели не распознан как JSON.")

desc_by_mod: dict[str, str] = st.session_state.get(_SK_MOD_LLM) or _default_mod_desc()

tabs = st.tabs([f"{m.key} · {m.short}" for m in MODULE_TABS])
for tab, mod in zip(tabs, MODULE_TABS):
    with tab:
        _render_module_tab(
            mod, wide_win, lsi_win, top_k, use_candles, desc_by_mod.get(mod.key)
        )
