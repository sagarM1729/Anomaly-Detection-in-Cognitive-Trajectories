"""Real-world dataset loader for cognitive trajectory analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import yaml


class RealDatasetLoader:
    """Load and normalize longitudinal cognitive assessments from real datasets."""

    def __init__(self, config_path: Optional[str] = None):
        with open(config_path or "config/config.yaml", "r") as fh:
            full_config = yaml.safe_load(fh)

        self.data_config = full_config.get("data", {})
        self.real_config = full_config.get("real_data", {})
        self.visit_months = sorted(self.data_config.get("visit_months", []))
        self.min_visits = self.data_config.get("min_visits_required", 2)
        self.summary: Dict[str, Any] = {}

    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load, harmonize, and filter the cognitive dataset."""
        dataset_path = Path(
            file_path or self.data_config.get("real_dataset_path", "")
        ).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Real dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path)
        original_rows = len(df)

        df = self._rename_columns(df)
        df = self._parse_visit_month(df)
        df = self._standardize_columns(df)
        df = self._deduplicate_visits(df)
        df = self._filter_patients(df)

        self._build_summary(df, dataset_path, original_rows)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            self.real_config.get("patient_id_col", "RID"): "patient_id",
            self.real_config.get("visit_code_col", "VISCODE"): "visit_code",
            self.real_config.get("visit_month_col", "Month"): "raw_month",
            self.real_config.get("exam_date_col", "EXAMDATE"): "exam_date",
            self.real_config.get("mmse_col", "MMSE"): "mmse",
            self.real_config.get("faq_col", "FAQ"): "faq",
            self.real_config.get("age_col", "AGE"): "age",
            self.real_config.get("gender_col", "PTGENDER"): "gender",
            self.real_config.get("education_col", "PTEDUCAT"): "education",
            self.real_config.get("apoe4_col", "APOE4"): "apoe4_copies",
            self.real_config.get("diagnosis_col", "DX"): "diagnosis",
            self.real_config.get("baseline_dx_col", "DX_bl"): "baseline_diagnosis",
        }

        existing_mapping = {k: v for k, v in mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)

        # Make sure key columns exist even if empty
        for required in ["patient_id", "visit_code", "mmse", "faq", "exam_date"]:
            if required not in df.columns:
                df[required] = np.nan

        extra_cols = self.real_config.get("extra_feature_columns", [])
        for col in extra_cols:
            if col in df.columns:
                df[col.lower()] = df[col]

        return df

    def _parse_visit_month(self, df: pd.DataFrame) -> pd.DataFrame:
        month_series = pd.Series([np.nan] * len(df), dtype=float)

        if "raw_month" in df.columns:
            month_series = pd.to_numeric(df["raw_month"], errors="coerce")

        missing_mask = month_series.isna()
        if missing_mask.any() and "visit_code" in df.columns:
            parsed = df.loc[missing_mask, "visit_code"].apply(self._viscode_to_month)
            month_series.loc[missing_mask] = parsed

        if month_series.isna().all():
            raise ValueError("Unable to determine visit months from dataset")

        month_series = month_series.astype(float)
        month_series = month_series.clip(lower=0)

        max_month = self.real_config.get("max_month")
        if max_month is not None:
            month_series = month_series.clip(upper=max_month)

        aligned_months = month_series.apply(self._align_to_schedule)
        df["visit_month"] = aligned_months
        return df

    def _viscode_to_month(self, code: Any) -> float:
        if pd.isna(code):
            return np.nan
        code_str = str(code).strip().lower()
        if not code_str:
            return np.nan

        direct_map = {
            "bl": 0,
            "sc": 0,
            "scmri": 0,
            "m00": 0,
            "m0": 0,
            "m06": 6,
            "m03": 3,
            "m12": 12,
            "m18": 18,
            "m24": 24,
            "m30": 30,
            "m36": 36,
            "m42": 42,
            "m48": 48,
            "m54": 54,
            "m60": 60,
            "y1": 12,
            "y2": 24,
            "y3": 36,
            "y4": 48,
            "y5": 60,
        }
        if code_str in direct_map:
            return direct_map[code_str]

        match = re.search(r"([my])(\d+)", code_str)
        if match:
            unit = match.group(1)
            value = int(match.group(2))
            if unit == "m":
                return float(value)
            if unit == "y":
                return float(value * 12)

        return np.nan

    def _align_to_schedule(self, month_value: float) -> float:
        if np.isnan(month_value) or not self.visit_months:
            return month_value
        schedule = np.array(self.visit_months)
        idx = (np.abs(schedule - month_value)).argmin()
        return float(schedule[idx])

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
        df = df.dropna(subset=["patient_id", "visit_month"])
        df["patient_id"] = df["patient_id"].astype(int)
        df["visit_month"] = df["visit_month"].astype(float)

        df["mmse"] = pd.to_numeric(df["mmse"], errors="coerce")
        df["faq"] = pd.to_numeric(df["faq"], errors="coerce")

        if "exam_date" in df.columns:
            df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce", format="%m-%d-%Y")

        if "gender" in df.columns:
            mapping = self.real_config.get("gender_mapping", {})
            df["gender"] = df["gender"].map(mapping).fillna(df["gender"].fillna("Unknown"))

        for col in ["education", "apoe4_copies", "age"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["pattern_type"] = df.get("pattern_type", "observed")
        df["pattern_type"] = df["pattern_type"].fillna("observed")

        return df

    def _deduplicate_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.sort_values(by=["patient_id", "visit_month", "exam_date"], kind="mergesort")

        def _select_best(group: pd.DataFrame) -> pd.Series:
            group = group.copy()
            completeness = (
                (~group["mmse"].isna()).astype(int)
                + (~group["faq"].isna()).astype(int)
            )
            group["_completeness"] = completeness
            group = group.sort_values(by=["_completeness", "exam_date"], ascending=[False, True])
            return group.iloc[0].drop("_completeness", errors="ignore")

        deduped = df.groupby(["patient_id", "visit_month"], as_index=False).apply(_select_best)
        deduped = deduped.reset_index(drop=True)
        return deduped

    def _filter_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        visit_counts = df.groupby("patient_id")["visit_month"].nunique()
        keep_ids = visit_counts[visit_counts >= self.min_visits].index
        filtered = df[df["patient_id"].isin(keep_ids)].copy()
        filtered = filtered.sort_values(["patient_id", "visit_month", "exam_date"], kind="mergesort")
        return filtered

    def _build_summary(self, df: pd.DataFrame, path: Path, original_rows: int) -> None:
        self.summary = {
            "data_path": str(path),
            "original_rows": original_rows,
            "rows_after_processing": len(df),
            "patients_total": int(df["patient_id"].nunique()),
            "mean_visits_per_patient": float(df.groupby("patient_id")["visit_month"].nunique().mean()),
            "visit_months_present": sorted(df["visit_month"].unique().tolist()),
            "mmse_missing_rate": float(df["mmse"].isna().mean()),
            "faq_missing_rate": float(df["faq"].isna().mean()),
        }

    def get_summary(self) -> Dict[str, Any]:
        return self.summary.copy()