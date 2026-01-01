from __future__ import annotations

import argparse
import json
import warnings
import time
from pathlib import Path
import sys
from typing import Any
import cProfile
import io
import pstats

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = Warning  # type: ignore[misc]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from app.main import (
    DATA_PATH,
    MODEL_PATH,
    ARTIFACTS_PATH,
    load_model,
    load_preprocessor,
    preprocess_input,
    new_features_creation,
    _apply_correlated_imputation,
    _ensure_required_columns,
    _validate_numeric_inputs,
    _validate_numeric_ranges,
)


def preprocess_input_legacy(df_raw: pd.DataFrame, artifacts) -> pd.DataFrame:
    df = df_raw.copy()

    for col in artifacts.required_input_columns:
        if col not in df.columns:
            df[col] = np.nan

    _ensure_required_columns(df, artifacts.required_input_columns)
    _validate_numeric_inputs(df, artifacts.numeric_required_columns)
    _validate_numeric_ranges(
        df,
        {k: v for k, v in artifacts.numeric_ranges.items() if k in artifacts.numeric_required_columns},
    )

    df["is_train"] = 0
    df["is_test"] = 1
    if "TARGET" not in df.columns:
        df["TARGET"] = 0

    df = new_features_creation(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in artifacts.columns_keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[artifacts.columns_keep]

    _apply_correlated_imputation(df, artifacts)

    for col, median in artifacts.numeric_medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(median)

    for col in artifacts.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    _ensure_required_columns(df, artifacts.required_input_columns)

    if "CODE_GENDER" in df.columns and (df["CODE_GENDER"] == "XNA").any():
        raise ValueError("CODE_GENDER cannot be 'XNA' based on training rules.")

    for col, max_val in artifacts.outlier_maxes.items():
        if col in df.columns and (df[col] >= max_val).any():
            raise ValueError(f"Input contains outlier values removed during training: {col}")

    df_hot = pd.get_dummies(df, columns=artifacts.categorical_columns)
    for col in artifacts.features_to_scaled:
        if col not in df_hot.columns:
            df_hot[col] = 0
    df_hot = df_hot[artifacts.features_to_scaled]

    scaled = artifacts.scaler.transform(df_hot)
    return pd.DataFrame(scaled, columns=artifacts.features_to_scaled, index=df.index)


def _load_input_sample(data_path: Path, columns: list[str], sample_size: int) -> pd.DataFrame:
    df = pd.read_parquet(data_path, columns=columns)
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df.reset_index(drop=True)


def _fill_required_inputs(df: pd.DataFrame, artifacts) -> pd.DataFrame:
    df_filled = df.copy()
    for col in artifacts.required_input_columns:
        if col not in df_filled.columns:
            df_filled[col] = np.nan
        if col in artifacts.numeric_medians:
            df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce").fillna(
                artifacts.numeric_medians[col]
            )
            if col in artifacts.numeric_ranges:
                min_val, max_val = artifacts.numeric_ranges[col]
                df_filled[col] = df_filled[col].clip(min_val, max_val)
        elif col in artifacts.categorical_columns:
            df_filled[col] = df_filled[col].fillna("Unknown")
        else:
            df_filled[col] = df_filled[col].fillna(0)
        if col in artifacts.outlier_maxes:
            max_val = artifacts.outlier_maxes[col]
            if pd.api.types.is_integer_dtype(df_filled[col]):
                replace_val = max_val - 1
            else:
                replace_val = np.nextafter(max_val, -np.inf)
            df_filled.loc[df_filled[col] >= max_val, col] = replace_val
    return df_filled


def _benchmark(
    *,
    name: str,
    preprocess_fn,
    model,
    artifacts,
    df_inputs: pd.DataFrame,
    batch_size: int,
    runs: int,
) -> dict[str, Any]:
    durations = []
    for _ in range(runs):
        for start in range(0, len(df_inputs), batch_size):
            batch = df_inputs.iloc[start:start + batch_size]
            t0 = time.perf_counter()
            features = preprocess_fn(batch, artifacts)
            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(features)[:, 1]
            else:
                _ = model.predict(features)
            durations.append((time.perf_counter() - t0) * 1000.0)
    durations = np.array(durations, dtype=float)
    return {
        "name": name,
        "batches": int(len(durations)),
        "batch_size": int(batch_size),
        "mean_ms": float(durations.mean()) if durations.size else 0.0,
        "p50_ms": float(np.percentile(durations, 50)) if durations.size else 0.0,
        "p95_ms": float(np.percentile(durations, 95)) if durations.size else 0.0,
        "throughput_rows_per_sec": float(
            (batch_size / (durations.mean() / 1000.0)) if durations.size else 0.0
        ),
    }


def _profile(preprocess_fn, model, artifacts, df_inputs: pd.DataFrame, batch_size: int) -> str:
    profiler = cProfile.Profile()
    batch = df_inputs.iloc[:batch_size]
    profiler.enable()
    features = preprocess_fn(batch, artifacts)
    if hasattr(model, "predict_proba"):
        _ = model.predict_proba(features)[:, 1]
    else:
        _ = model.predict(features)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile and benchmark inference latency.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--artifacts-path", type=Path, default=ARTIFACTS_PATH)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=Path("docs/performance/benchmark_results.json"))
    parser.add_argument("--output-profile", type=Path, default=Path("docs/performance/profile_summary.txt"))
    args = parser.parse_args()

    preprocessor = load_preprocessor(args.data_path, args.artifacts_path)
    model = load_model(args.model_path)

    input_cols = list(preprocessor.required_input_columns)
    df_inputs = _load_input_sample(args.data_path, input_cols, args.sample_size)
    df_inputs = _fill_required_inputs(df_inputs, preprocessor)

    results = []
    results.append(
        _benchmark(
            name="optimized_preprocess",
            preprocess_fn=preprocess_input,
            model=model,
            artifacts=preprocessor,
            df_inputs=df_inputs,
            batch_size=args.batch_size,
            runs=args.runs,
        )
    )
    results.append(
        _benchmark(
            name="legacy_preprocess_alignment",
            preprocess_fn=preprocess_input_legacy,
            model=model,
            artifacts=preprocessor,
            df_inputs=df_inputs,
            batch_size=args.batch_size,
            runs=args.runs,
        )
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    profile_text = _profile(preprocess_input, model, preprocessor, df_inputs, args.batch_size)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(profile_text, encoding="utf-8")

    print(f"Saved benchmarks to {args.output_json}")
    print(f"Saved profile to {args.output_profile}")


if __name__ == "__main__":
    main()
