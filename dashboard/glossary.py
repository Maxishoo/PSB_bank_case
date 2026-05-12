from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    id: str
    public_name: str
    formula: str
    meaning: str


STAT_LABELS_FOR_LLM: dict[str, str] = {
    "current": "сейчас",
    "median": "типичный уровень за последний год (~252 торговых дня)",
    "delta_7d": "изменение за 7 дней",
    "delta_30d": "изменение за 30 дней",
}


TERMS: dict[str, Term] = {
    "anomaly_score": Term(
        id="anomaly_score",
        public_name="насколько значение необычное",
        formula="(сегодня − типичный уровень) ÷ (1,4826 × типичный разброс ряда), по скользящему окну ~3 года",
        meaning=(
            "Показывает, насколько сегодняшнее значение выбивается из «обычного для него» хода. "
            "Ноль — как обычно; чем дальше от нуля по модулю, тем реже такое значение. "
            "В коде данных это называется MAD-score."
        ),
    ),
    "typical_level_year": Term(
        id="typical_level_year",
        public_name="типичный уровень за последний год",
        formula="медиана последних ~252 торговых дней по этому показателю",
        meaning="Середина «нормального» хода за год: половина дней была выше, половина ниже.",
    ),
    "change_7d": Term(
        id="change_7d",
        public_name="изменение за 7 дней",
        formula="значение сегодня − значение 7 торговых дней назад",
        meaning="Быстро показывает, вырос показатель или упал за неделю.",
    ),
    "change_30d": Term(
        id="change_30d",
        public_name="изменение за 30 дней",
        formula="значение сегодня − значение 30 торговых дней назад",
        meaning="Сглаженнее недельного: ловит тренд за месяц.",
    ),
    "lsi_change_7d": Term(
        id="lsi_change_7d",
        public_name="насколько вырос или упал LSI за неделю",
        formula="LSI сегодня − LSI 7 дней назад",
        meaning="Положительное число — индекс стресса выше, чем неделю назад; отрицательное — ниже.",
    ),
    "lsi_change_30d": Term(
        id="lsi_change_30d",
        public_name="насколько вырос или упал LSI за месяц",
        formula="LSI сегодня − LSI 30 дней назад",
        meaning="То же, что недельное изменение, но на горизонте месяца.",
    ),
    "place_in_year": Term(
        id="place_in_year",
        public_name="место среди прошлого года",
        formula="доля дней за год, когда показатель был не выше сегодняшнего (от 0 до 100%)",
        meaning=(
            "100% — сегодня максимум за год по этому ряду, 0% — минимум, 50% — ровно середина. "
            "В матстатистике это близко к перцентилю ранга."
        ),
    ),
    "typical_corridor": Term(
        id="typical_corridor",
        public_name="типичный коридор на графике",
        formula="полоса от 25%-го до 75%-го уровня в выбранном окне дней",
        meaning="Где лежит «обычная» половина значений без выбросов; помогает глазами отделить шум от выброса.",
    ),
    "typical_line_window": Term(
        id="typical_line_window",
        public_name="типичный уровень в окне графика",
        formula="медиана значений на отрезке ±90 дней вокруг выбранной даты",
        meaning="Пунктир на мини-графике: ориентир «середины» за соседние дни.",
    ),
    "forecast_spread": Term(
        id="forecast_spread",
        public_name="разброс прогноза",
        formula="сглаженный LSI ± одно стандартное отклонение сырого прогноза за 21 день",
        meaning="Полоса вокруг линии: чем шире, тем менее уверена модель в уровне в эти дни.",
    ),
    "global_lsi": Term(
        id="global_lsi",
        public_name="глобальный LSI",
        formula="модель 0–100 по всей доступной истории рынка",
        meaning="Оценка напряжённости ликвидности сейчас относительно всей накопленной истории наблюдений.",
    ),
    "local_lsi": Term(
        id="local_lsi",
        public_name="локальный LSI",
        formula="модель 0–100 на скользящем окне ~252 торговых дня (три подмодели 90/180/365 дн., затем среднее)",
        meaning="Оценка относительно текущего рыночного режима: насколько сегодня необычно на горизонте последнего года, слабее тянет за собой многолетний тренд.",
    ),
    "together_with_lsi": Term(
        id="together_with_lsi",
        public_name="насколько ряд движется вместе с LSI",
        formula="число от −1 до +1: насколько два ряда похожи по направлению движения по всем дням на графике",
        meaning=(
            "Ближе к +1 — фактор и индекс чаще растут или падают вместе; ближе к −1 — двигаются навстречу; "
            "около 0 — почти не связаны. В учебниках это называют корреляцией (r)."
        ),
    ),
    "bar_strength_sum": Term(
        id="bar_strength_sum",
        public_name="суммарная сила сигналов по модулю",
        formula="сумма модулей оценок необычности по топ-драйверам внутри модуля",
        meaning="Чем выше столбик, тем больше суммарное «выбивание из нормы» по этому модулю в выбранный день.",
    ),
    "ruonia": Term(
        id="ruonia",
        public_name="RUONIA",
        formula="средневзвешенная процентная ставка по овернайт-репо в рублях на Московской бирже (однодневные сделки)",
        meaning=(
            "Ориентир стоимости однодневной межбанковской ликвидности в рублях: выше обычного — "
            "деньги на «овернайт» дороже, чаще дефицит; ниже — избыток рублёвой ликвидности на коротком конце."
        ),
    ),
    "required_reserves": Term(
        id="required_reserves",
        public_name="обязательные резервы",
        formula="нормативный объём средств на корсчетах в ЦБ, который банк обязан удерживать (зависит от обязательств)",
        meaning="Буфер ликвидности регуляторного требования: сравнение факта с нормативом показывает, «копит» ли система запас или живёт впритык.",
    ),
    "cbr_repo": Term(
        id="cbr_repo",
        public_name="репо ЦБ",
        formula="кредитование банков у Банка России под залог ценных бумаг (аукционы репо)",
        meaning="Инструмент дневного/краткого фондирования: высокий спрос — банкам не хватает рублей на рынке, они идут к регулятору.",
    ),
    "ofz": Term(
        id="ofz",
        public_name="ОФЗ",
        formula="облигации федерального займа — долговые бумаги Минфина РФ в рублях",
        meaning="Государственный долг в удобной для рынка форме: спрос на ОФЗ часто отражает «запасной аэродром» для свободной ликвидности.",
    ),
    "cover_ratio": Term(
        id="cover_ratio",
        public_name="cover ratio",
        formula="отношение суммы заявок к объёму размещения на аукционе (сколько раз «перекрыли» выпуск)",
        meaning="Выше 1 — заявок больше, чем бумаг; сильный переспрос — рынок готов забрать весь объём и ещё. Ниже 1 — недоспрос.",
    ),
    "cover_ratio_cbr_repo": Term(
        id="cover_ratio_cbr_repo",
        public_name="cover ratio репо ЦБ",
        formula="перекрытие заявок на аукционе репо Банка России относительно предложенного объёма",
        meaning="Показывает, насколько банки «просят» рубли у ЦБ под залог: высокий cover — очередь за фондированием у регулятора.",
    ),
    "cover_ratio_ofz": Term(
        id="cover_ratio_ofz",
        public_name="cover ratio ОФЗ",
        formula="перекрытие заявок на аукционе ОФЗ относительно объёма выпуска",
        meaning="Спрос на госдолг в момент размещения: высокий cover — ликвидность ищет «якорь» в ОФЗ; низкий — осторожность по длинному долгу.",
    ),
    "reserve_fact_need_spread": Term(
        id="reserve_fact_need_spread",
        public_name="спред усреднения резервов",
        formula="фактический остаток средств на корсчёте − нормативная потребность (обязательные резервы), в денежных единицах",
        meaning="Положительный спред — избыток относительно нормы (часто «запас на ветер»); около нуля или отрицательный — мало буфера.",
    ),
    "repo_rate_vs_key": Term(
        id="repo_rate_vs_key",
        public_name="спред ставки репо к ключевой",
        formula="ставка отсечения (или итог аукциона) репо ЦБ минус ключевая ставка Банка России, в п.п.",
        meaning="Насколько дорого обходится краткое фондирование у регулятора относительно «якорной» ключевой ставки: ближе к верху коридора — напряжение.",
    ),
    "weekly_tax_intensity": Term(
        id="weekly_tax_intensity",
        public_name="налоговая нагрузка недели",
        formula="агрегированный вес налоговых выплат клиентов в календаре на ближайшие дни (модельный индекс)",
        meaning="Тяжёлая налоговая неделя — отток средств со счетов на уплату налогов, давление на ликвидность банковской системы.",
    ),
    "structural_liquidity_gap": Term(
        id="structural_liquidity_gap",
        public_name="структурный дефицит/профицит ликвидности (ЦБ)",
        formula="оценка Банка России: насколько система в целом должна/может брать или размещать рубли у ЦБ (структурный баланс)",
        meaning="Показывает «базовый» разрыв ликвидности без сиюминутных колебаний: дефицит — системе не хватает рублей на постоянной основе.",
    ),
    "treasury_daily_flow": Term(
        id="treasury_daily_flow",
        public_name="давление казначейства (изменение за день)",
        formula="изменение остатков/потоков средств федерального казначейства на корсчетах за торговый день",
        meaning="Резкий отток казначейства — деньги уходят с корсчетов банков; приток — возвращаются на счета, смягчая дефицит.",
    ),
    "tax_calendar_m4": Term(
        id="tax_calendar_m4",
        public_name="налоги (календарь)",
        formula="модельные веса и типы налоговых дат в горизонте нескольких недель",
        meaning="Когда клиенты массово платят налоги, на корсчетах банков меньше свободных рублей — это отдельный канал давления на ликвидность.",
    ),
    "seasonality_m4": Term(
        id="seasonality_m4",
        public_name="сезонность",
        formula="повторяющиеся в календаре паттерны (отчётность, квартальные пики и т.п.)",
        meaning="Учитывает, что часть напряжения ликвидности повторяется из года в год в одни и те же периоды.",
    ),
    "federal_treasury": Term(
        id="federal_treasury",
        public_name="федеральное казначейство",
        formula="система счетов и платежей бюджета: остатки и операции казначейства на рынке",
        meaning="Притоки и оттоки бюджетных средств напрямую меняют объём рублёвой ликвидности у банков.",
    ),
}


INLINE_HINT_NEEDLES: tuple[tuple[str, str, str | None], ...] = tuple(
    sorted(
        (
            ("Структурный дефицит ликвидности (ЦБ)", "structural_liquidity_gap", None),
            ("Давление казначейства (изменение за день)", "treasury_daily_flow", None),
            ("Cover ratio репо ЦБ", "cover_ratio_cbr_repo", None),
            ("Cover ratio ОФЗ", "cover_ratio_ofz", None),
            ("Спред ставки репо к ключевой", "repo_rate_vs_key", None),
            ("Спред усреднения резервов", "reserve_fact_need_spread", None),
            ("Налоговая нагрузка недели", "weekly_tax_intensity", None),
            ("RUONIA", "ruonia", "RUONIA"),
            ("Cover ratio", "cover_ratio", "Cover ratio"),
            ("ОФЗ", "ofz", "ОФЗ"),
            ("репо ЦБ", "cbr_repo", None),
        ),
        key=lambda x: -len(x[0]),
    )
)


def term_title(term_id: str) -> str:
    """Короткая строка для title у summary (подсказка при наведении)."""
    t = TERMS[term_id]
    return html.escape(f"Нажмите, чтобы развернуть. {t.formula}", quote=True)


def term_abbr(term_id: str, *, label: str | None = None) -> str:
    """Раскрывающийся блок: клик по термину показывает формулу и смысл (unsafe_allow_html).

    У обычного <abbr title> на многих системах только курсор «?» при наведении, по клику ничего
    не происходит — поэтому используем <details>/<summary>.
    """
    t = TERMS[term_id]
    inner = html.escape(label if label is not None else t.public_name)
    formula_e = html.escape(t.formula)
    meaning_e = html.escape(t.meaning)
    tip = term_title(term_id)
    return (
        f'<details class="lsi-term"><summary title="{tip}">{inner}</summary>'
        f'<div class="lsi-term-body"><strong>Формула:</strong> {formula_e}<br>'
        f'<strong>Смысл:</strong> {meaning_e}</div></details>'
    )


def inline_hints(text: str | None) -> str:
    """Подставляет раскрывающиеся подсказки для известных подстрок (названия драйверов, пояснения)."""
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""
    needles = INLINE_HINT_NEEDLES
    out: list[str] = []
    i = 0
    while i < len(s):
        hit: tuple[str, str, str | None] | None = None
        for needle, tid, lbl in needles:
            if s.startswith(needle, i):
                hit = (needle, tid, lbl)
                break
        if hit:
            needle, tid, lbl = hit
            out.append(term_abbr(tid, label=lbl or needle))
            i += len(needle)
        else:
            out.append(html.escape(s[i]))
            i += 1
    return "".join(out)


def module_label_html(mod_key: str) -> str:
    """Заголовок карточки модуля M1…M5 с подсказками (совпадает по смыслу с MOD_LABELS на главной)."""
    key = (mod_key or "").lower()
    if key == "m1":
        return (
            "M1 · "
            f"{term_abbr('required_reserves', label='Резервы')} / "
            f"{term_abbr('ruonia', label='RUONIA')}"
        )
    if key == "m2":
        return f"M2 · {term_abbr('cbr_repo', label='Репо ЦБ')}"
    if key == "m3":
        return f"M3 · {term_abbr('ofz', label='ОФЗ')}"
    if key == "m4":
        return (
            "M4 · "
            f"{term_abbr('tax_calendar_m4', label='Налоги')} / "
            f"{term_abbr('seasonality_m4', label='сезонность')}"
        )
    if key == "m5":
        return f"M5 · {term_abbr('federal_treasury', label='Казначейство')}"
    return html.escape(mod_key)


def metric_help(term_id: str) -> str:
    """Текст для st.metric(..., help=...)."""
    t = TERMS[term_id]
    return f"{t.formula}\n\n{t.meaning}"


def llm_lexicon_block() -> str:
    """Вставляется в user-prompt: как формулировать ответы."""
    lines = [
        "ЛЕКСИКОН ДЛЯ ТЕКСТА ПОЛЬЗОВАТЕЛЮ (используй эти формулировки, избегай жаргона без перевода):",
    ]
    for t in TERMS.values():
        chunk = t.meaning.split("В коде")[0].strip()
        lines.append(f"- «{t.public_name}» — {chunk}")
    lines.append(
        "Не пиши пользователю сырые имена колонок (m2_…, MAD-score как термин). "
        "Можно один раз в скобках пояснить «внутри модели это робастная z-оценка», если нужно."
    )
    return "\n".join(lines)


def render_glossary_llm_expander(*, api_ok: bool, llm_chat, system_prompt: str) -> None:
    """Expander: выбор термина, краткое описание, вопрос к LLM."""
    import streamlit as st

    with st.expander("Справка по терминам и вопрос к LLM", expanded=False):
        st.caption(
            "На странице **подчёркнутые термины** (в т.ч. RUONIA, ОФЗ, Cover ratio в названиях факторов "
            "и заголовках M1–M5) — нажмите, чтобы развернуть формулу и смысл. "
            "Здесь тот же полный список и вопрос к LLM."
        )
        term_ids = list(TERMS.keys())
        choice = st.selectbox(
            "Термин",
            term_ids,
            format_func=lambda tid: TERMS[tid].public_name,
            key="glossary_term_pick",
        )
        t = TERMS[choice]
        st.markdown(f"**Коротко:** {t.public_name}")
        st.markdown(f"**Формула:** {t.formula}")
        st.markdown(f"**Смысл:** {t.meaning}")
        follow = st.text_input(
            "Дополнительный вопрос (по желанию)",
            placeholder="Например: при каких значениях это опасно для ликвидности?",
            key="glossary_llm_followup",
        )
        if st.button("Спросить LLM подробнее", key="glossary_llm_ask", width="stretch"):
            if not api_ok:
                st.warning("Задайте API-ключ в «⚙️ Настройки».")
            else:
                user = (
                    f"Объясни простым языком для сотрудника казначейства банка термин «{t.public_name}».\n\n"
                    f"В дашборде он описан так:\nФормула: {t.formula}\nСмысл: {t.meaning}\n\n"
                )
                if follow.strip():
                    user += f"Дополнительный вопрос пользователя: {follow.strip()}\n\n"
                user += (
                    "Ответ: 6–12 предложений по-русски, без выдуманных цифр и без новых формул, "
                    "если они не следуют из описания выше. Можно один пример интерпретации «если число большое»."
                )
                with st.spinner("LLM…"):
                    raw = llm_chat(user, system_prompt, False)
                if raw:
                    st.markdown(raw)
