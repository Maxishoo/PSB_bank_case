from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_wide_lsi(df: pd.DataFrame, dashboard_dir: str | Path | None = None) -> Path:
    if dashboard_dir is None:
        dashboard_dir = Path.cwd()
        if dashboard_dir.name != "dashboard":
            cand = dashboard_dir / "dashboard"
            if cand.is_dir():
                dashboard_dir = cand
    dashboard_dir = Path(dashboard_dir)
    data_dir = dashboard_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_csv = data_dir / "wide_lsi.csv"
    df.to_csv(out_csv, index=False)
    print(f"[export] CSV: {out_csv}  ({df.shape[0]} строк, {df.shape[1]} колонок)")

    try:
        out_pq = data_dir / "wide_lsi.parquet"
        df.to_parquet(out_pq, index=False)
        print(f"[export] Parquet: {out_pq}")
    except Exception as e:
        print(f"[export] Parquet пропущен: {e}")

    return out_csv


if __name__ == "__main__":
    print("Импортируйте `export_wide_lsi(wide_lsi)` в ноутбуке после обучения моделей.")
