# construire drift avec evidently

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional plotting dependency
    raise SystemExit(
        "matplotlib is required for plots. Install it with: pip install matplotlib"
    ) from exc


DEFAULT_FEATURES = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "CODE_GENDER",
    "DAYS_EMPLOYED",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "FLAG_OWN_CAR",
]

CATEGORICAL_FEATURES = {"CODE_GENDER", "FLAG_OWN_CAR"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value)


def _load_logs(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries: list[dict[str, object]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        return pd.DataFrame(), pd.DataFrame()
    inputs = [entry.get("inputs", {}) for entry in entries if entry.get("inputs")]
    inputs_df = pd.DataFrame.from_records(inputs)
    meta_df = pd.DataFrame.from_records(entries)
    return inputs_df, meta_df


def _psi(reference: pd.Series, production: pd.Series, eps: float = 1e-6) -> float:
    ref_freq = reference.value_counts(normalize=True, dropna=False)
    prod_freq = production.value_counts(normalize=True, dropna=False)
    categories = ref_freq.index.union(prod_freq.index)
    ref_probs = ref_freq.reindex(categories, fill_value=0).to_numpy()
    prod_probs = prod_freq.reindex(categories, fill_value=0).to_numpy()
    ref_probs = np.clip(ref_probs, eps, None)
    prod_probs = np.clip(prod_probs, eps, None)
    return float(np.sum((ref_probs - prod_probs) * np.log(ref_probs / prod_probs)))


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _plot_numeric(ref: pd.Series, prod: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(ref.dropna(), bins=30, alpha=0.6, label="reference")
    plt.hist(prod.dropna(), bins=30, alpha=0.6, label="production")
    plt.title(f"Distribution: {ref.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_categorical(ref: pd.Series, prod: pd.Series, output_path: Path, max_categories: int = 10) -> None:
    ref_series = ref.fillna("Unknown")
    prod_series = prod.fillna("Unknown")
    top = ref_series.value_counts().index[:max_categories]
    ref_series = ref_series.where(ref_series.isin(top), other="__OTHER__")
    prod_series = prod_series.where(prod_series.isin(top), other="__OTHER__")
    ref_freq = ref_series.value_counts(normalize=True)
    prod_freq = prod_series.value_counts(normalize=True)
    plot_df = pd.DataFrame({"reference": ref_freq, "production": prod_freq}).fillna(0)
    plot_df.sort_values("reference", ascending=False).plot(kind="bar", figsize=(7, 4))
    plt.title(f"Distribution: {ref.name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_score_distribution(scores: pd.Series, output_path: Path, bins: int = 30) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(scores.dropna(), bins=bins, range=(0, 1), alpha=0.8, color="#4C78A8")
    plt.title("Prediction score distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_prediction_rate(predictions: pd.Series, output_path: Path) -> None:
    counts = predictions.value_counts(normalize=True, dropna=False).sort_index()
    plt.figure(figsize=(4, 4))
    plt.bar(counts.index.astype(str), counts.values, color="#F58518")
    plt.title("Prediction rate")
    plt.xlabel("Predicted class")
    plt.ylabel("Share")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(
    log_path: Path,
    reference_path: Path,
    output_dir: Path,
    sample_size: int,
    psi_threshold: float,
    score_bins: int,
) -> Path:
    inputs_df, meta_df = _load_logs(log_path)
    if inputs_df.empty:
        raise SystemExit(f"No inputs found in logs: {log_path}")

    features = [col for col in DEFAULT_FEATURES if col in inputs_df.columns]
    if not features:
        raise SystemExit("No matching features found in production logs.")

    reference_df = pd.read_parquet(reference_path, columns=features)
    if sample_size and len(reference_df) > sample_size:
        reference_df = reference_df.sample(sample_size, random_state=42)

    numeric_features = [col for col in features if col not in CATEGORICAL_FEATURES]
    production_df = _coerce_numeric(inputs_df, numeric_features)
    reference_df = _coerce_numeric(reference_df, numeric_features)

    summary_rows: list[dict[str, object]] = []
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for feature in features:
        if feature not in reference_df.columns:
            continue
        ref_series = reference_df[feature]
        prod_series = production_df[feature]
        if feature in CATEGORICAL_FEATURES:
            psi_value = _psi(ref_series, prod_series)
            summary_rows.append(
                {
                    "feature": feature,
                    "type": "categorical",
                    "psi": round(psi_value, 4),
                    "drift_detected": psi_value >= psi_threshold,
                }
            )
            plot_path = plots_dir / f"{_safe_name(feature)}.png"
            _plot_categorical(ref_series, prod_series, plot_path)
        else:
            ref_clean = ref_series.dropna()
            prod_clean = prod_series.dropna()
            if ref_clean.empty or prod_clean.empty:
                continue
            stat, pvalue = stats.ks_2samp(ref_clean, prod_clean)
            summary_rows.append(
                {
                    "feature": feature,
                    "type": "numeric",
                    "ks_stat": round(float(stat), 4),
                    "p_value": round(float(pvalue), 6),
                    "drift_detected": pvalue < 0.05,
                }
            )
            plot_path = plots_dir / f"{_safe_name(feature)}.png"
            _plot_numeric(ref_series, prod_series, plot_path)

    summary_df = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_report.html"

    total_calls = len(meta_df)
    error_rate = float((meta_df.get("status_code", pd.Series(dtype=int)) >= 400).mean()) if total_calls else 0.0
    latency_ms = meta_df.get("latency_ms", pd.Series(dtype=float)).dropna()
    latency_p50 = float(latency_ms.quantile(0.5)) if not latency_ms.empty else 0.0
    latency_p95 = float(latency_ms.quantile(0.95)) if not latency_ms.empty else 0.0

    valid_meta = meta_df
    if "status_code" in meta_df.columns:
        valid_meta = meta_df[meta_df["status_code"] < 400]
    score_series = (
        pd.to_numeric(valid_meta.get("probability", pd.Series(dtype=float)), errors="coerce")
        .dropna()
    )
    pred_series = (
        pd.to_numeric(valid_meta.get("prediction", pd.Series(dtype=float)), errors="coerce")
        .dropna()
    )

    score_metrics_html = "<li>No prediction scores available.</li>"
    score_plots_html = ""
    if not score_series.empty:
        score_mean = float(score_series.mean())
        score_p50 = float(score_series.quantile(0.5))
        score_p95 = float(score_series.quantile(0.95))
        score_min = float(score_series.min())
        score_max = float(score_series.max())
        score_metrics = [
            f"<li>Score mean: {score_mean:.4f}</li>",
            f"<li>Score p50: {score_p50:.4f}</li>",
            f"<li>Score p95: {score_p95:.4f}</li>",
            f"<li>Score min: {score_min:.4f}</li>",
            f"<li>Score max: {score_max:.4f}</li>",
        ]
        score_metrics_html = "\n".join(score_metrics)
        score_plot_path = plots_dir / "score_distribution.png"
        _plot_score_distribution(score_series, score_plot_path, bins=score_bins)
        score_plots_html = "<img src='plots/score_distribution.png' />"

    if not pred_series.empty:
        pred_rate = float(pred_series.mean())
        score_metrics_html += f"\n<li>Predicted default rate: {pred_rate:.2%}</li>"
        pred_plot_path = plots_dir / "prediction_rate.png"
        _plot_prediction_rate(pred_series, pred_plot_path)
        score_plots_html += "\n<img src='plots/prediction_rate.png' />"

    summary_html = summary_df.to_html(index=False, escape=False)
    plots_html = "\n".join(
        f"<h4>{row['feature']}</h4><img src='plots/{_safe_name(row['feature'])}.png' />"
        for _, row in summary_df.iterrows()
    )

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Drift Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 8px; }}
      th {{ background: #f3f3f3; }}
      img {{ max-width: 720px; }}
    </style>
  </head>
  <body>
    <h2>Production Monitoring Summary</h2>
    <ul>
      <li>Total calls: {total_calls}</li>
      <li>Error rate: {error_rate:.2%}</li>
      <li>Latency p50: {latency_p50:.2f} ms</li>
      <li>Latency p95: {latency_p95:.2f} ms</li>
    </ul>
    <h2>Score Monitoring</h2>
    <ul>
      {score_metrics_html}
    </ul>
    {score_plots_html}
    <h2>Data Drift Summary</h2>
    {summary_html}
    <h2>Feature Distributions</h2>
    {plots_html}
  </body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a drift report from production logs.")
    parser.add_argument("--logs", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--reference", type=Path, default=Path("data/data_final.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--score-bins", type=int, default=30)
    args = parser.parse_args()

    report_path = generate_report(
        log_path=args.logs,
        reference_path=args.reference,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        psi_threshold=args.psi_threshold,
        score_bins=args.score_bins,
    )
    print(f"Drift report saved to {report_path}")


if __name__ == "__main__":
    main()
