# LSI Dashboard (Streamlit)

Интерактивный дашборд по системе раннего предупреждения стресса ликвидности рублёвого денежного рынка (см. `тз.docx`).

## Структура

```
dashboard/
├─ app.py                         # Главная: глобальный + локальный LSI, драйверы с мини-графиками, LLM-разбор
├─ pages/
│  ├─ 01_📊_Модули.py             # Графики по M1, M2, M3, M4, M5 — связь фич с LSI
│  └─ 02_🤖_Аналитик.py           # Чат-RAG строго по таблицам системы (DeepSeek LLM)
├─ utils.py                        # Загрузка данных (lsi_panel, lsi_panel_local, wide_lsi), алерты
├─ export_from_notebook.py         # Экспорт `wide_lsi` из ноутбука (легаси-хелпер)
├─ requirements.txt
└─ data/                           # сюда кладётся wide_lsi.csv / lsi_panel.csv / lsi_panel_local.csv / …
```

## Установка

```bash
cd /Users/mayadzyuba/tryM4
python -m venv .venv && source .venv/bin/activate   # если ещё нет
pip install -r dashboard/requirements.txt
```

## Экспорт данных из ноутбука

В `final.ipynb` после обучения моделей (где формируется `wide_lsi` со столбцами `LSI_*`) добавьте ячейку:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "dashboard"))
from export_from_notebook import export_wide_lsi

export_wide_lsi(wide_lsi)   # запишет dashboard/data/wide_lsi.csv (+ parquet)
```

## Запуск

```bash
cd /Users/mayadzyuba/tryM4
streamlit run dashboard/app.py
```

По умолчанию приложение откроется на `http://localhost:8501`.

## Что есть

- **Главная (`app.py`)** — два главных графика LSI с кликом для выбора даты:
  - **Глобальный LSI** (`LSI_lgbm_tuned`) — абсолютный уровень стресса 0–100 относительно всей истории.
  - **Локальный LSI** (`LSI_lgbm_local_multi`, multi-window 90/180/365 д.) — перцентиль относительно последних 12 мес.
  Под графиками — секция «Что давит на LSI»: для каждого активного драйвера M1–M5 показывается мини-график значения за ±90 дней с подсветкой выбранной даты, текущее значение, медиана за год, Δ за 7/30 дней и плейн-человеческое объяснение. Плюс LLM-разбор и stacked-area вкладов модулей.
- **Модули (`01_📊_Модули.py`)** — для каждого из M1–M5 — топ-фичи с наибольшей корреляцией с LSI, временные ряды.
- **Аналитик (`02_🤖_Аналитик.py`)** — отдельный чат-интерфейс с RAG **строго по таблицам системы**. Парсит период из запроса («март 2022», «август 2023», «последний год», «2022 год», «с марта по июнь 2022», ISO/русские даты), классифицирует интент (период / top-stress / why), детерминированно строит числовой контекст из `wide_lsi.csv` + `lsi_panel.csv` + `lsi_panel_local.csv` + `stress_events.csv` + налоговый календарь + статические описания модулей и отдаёт LLM с жёстким system-промптом «использовать только данные из `<DATA_CONTEXT>`, не выдумывать». История диалога сохраняется в `session_state`, есть кнопка «Очистить», есть expander «Сырой контекст», есть готовые примеры из ТЗ. Если за период данных нет — отвечает «нет данных в системе».

  **DeepSeek (по умолчанию, если задан ключ):**
  ```bash
  export DEEPSEEK_API_KEY=sk-...
  export DEEPSEEK_MODEL=deepseek-chat        # или deepseek-reasoner
  # export DEEPSEEK_BASE_URL=https://api.deepseek.com/v1   # дефолт
  ```

  **OpenAI:**
  ```bash
  export OPENAI_API_KEY=sk-...
  export OPENAI_MODEL=gpt-4o-mini
  ```

  Если заданы оба ключа, можно явно выбрать: `export LLM_PROVIDER=deepseek` (или `openai`).

## Пороговые значения алертов

Согласно ТЗ:

- 🟢 Зелёный — `0 ≤ LSI < 40`
- 🟡 Жёлтый — `40 ≤ LSI < 70`
- 🔴 Красный — `LSI ≥ 70`

Изменить можно в `dashboard/utils.py` → `LSI_THRESHOLDS`.

## Какую колонку LSI берём как «итоговую»

Дашборд автоматически выбирает первую доступную:

1. `LSI_lgbm_tuned_tax_adj`
2. `LSI_lgbm_tuned`
3. `LSI_ensemble_tax_adj`
4. `LSI_ensemble`
5. `LSI_lgbm_huber`
6. `LSI_teacher` (fallback)
