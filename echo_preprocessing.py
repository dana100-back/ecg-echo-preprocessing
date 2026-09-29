"""Illustrative echo-report preprocessing adapted from research working notebooks.

Extraction rules depend on the report template and are not clinically validated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("환자번호", "생년월일", "검사시행일", "검사결과")
NUMBER = r"(-?\d+(?:\.\d+)?)"

# Match labels and values on the same line; keep E' separate from E/E' ratios.
PATTERNS = {
    "ef": r"\b(?:LVEF|EF)\b[^\d\n]{0,20}" + NUMBER,
    "lvesd": r"\b(?:LVID\s*s|LVESD)\b[^\d\n]{0,20}" + NUMBER,
    "lvedd": r"\b(?:LVID\s*d|LVEDD)\b[^\d\n]{0,20}" + NUMBER,
    "la_size": r"(?:\bLA\s*size\b|\bLA\s*\(\s*M[- ]?mode\s*\))[^\d\n]{0,20}" + NUMBER,
    "la_volume": r"\bLA\s*vol(?:ume)?\b[^\d\n]{0,20}" + NUMBER,
    "lavi": r"\bLAVI\b[^\d\n]{0,20}" + NUMBER,
    "dt": r"\bDT\b[^\d\n]{0,20}" + NUMBER,
    "e": r"\bMV\s+E\s+V\b[^\d\n]{0,20}" + NUMBER,
    "a": r"\bMV\s+A\b(?!\s*['’/])[^\d\n]{0,20}" + NUMBER,
    "s": r"\bMV\s+S/[^\d\n]{0,20}" + NUMBER,
    "e_prime": r"(?<!/)\b(?:MV\s+)?E\s*['’](?!\s*/)[^\d\n]{0,20}" + NUMBER,
    "a_prime": r"(?<!/)\b(?:MV\s+)?A\s*['’](?!\s*/)[^\d\n]{0,20}" + NUMBER,
    "s_prime": r"(?<!/)\b(?:MV\s+)?S\s*['’](?!\s*/)[^\d\n]{0,20}" + NUMBER,
}
COMPILED = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PATTERNS.items()}
FEATURE_COLUMNS = list(PATTERNS) + ["rwma_mentioned", "afib_during_exam_mentioned"]
FEATURE_ALIASES = {
    "EF": "ef", "LVESD": "lvesd", "LVESD_val": "lvesd",
    "LVEDD": "lvedd", "LVEDD_val": "lvedd",
    "LA size": "la_size", "LA_size": "la_size", "LA_size_val": "la_size",
    "LA volume": "la_volume", "LA_volume": "la_volume", "LA_volume_val": "la_volume",
    "LAVI": "lavi", "LAVI_val": "lavi", "DT": "dt",
    "E": "e", "A": "a", "S": "s", "E'": "e_prime", "A'": "a_prime", "S'": "s_prime",
    "E_prime": "e_prime", "A_prime": "a_prime", "S_prime": "s_prime",
    "RWMA_mentioned": "rwma_mentioned",
}
DEMOGRAPHIC_COLUMNS = {
    "생년월일": "생년월일",
    "성별": "성별",
    "신장(cm)": "신장",
    "체중(kg)": "체중",
    "BMI(kg/m²)": "BMI",
    "검사처방일": "검사처방일",
}


def extract_measurements(report: str) -> dict:
    """Extract the first value for each recognized label in a report."""
    values = {
        name: float(match.group(1)) if (match := regex.search(report)) else float("nan")
        for name, regex in COMPILED.items()
    }
    values["rwma_mentioned"] = bool(re.search(r"\b(?:hypokinesia|akinesia|dyskinesia)\b", report, re.I))
    values["afib_during_exam_mentioned"] = any(
        re.search(r"\b(?:afib|a[- ]?fib|atrial fibrillation)\b", line, re.I)
        and re.search(r"\bduring\b", line, re.I)
        for line in report.splitlines()
    )
    return values


def standardize_measurement_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy measurement names and convert extracted values to numbers.

    This expects numeric values or individual measurement strings, such as
    'EF: 55 %', rather than a complete report containing multiple measurements.
    """
    result = table.copy()
    for old, canonical in FEATURE_ALIASES.items():
        if old in result:
            if canonical in result:
                result[canonical] = result[canonical].combine_first(result[old])
                result = result.drop(columns=old)
            else:
                result = result.rename(columns={old: canonical})
    for column in PATTERNS:
        if column not in result:
            result[column] = pd.Series(index=result.index, dtype="Float64")
        else:
            text = result[column].astype("string")
            numbers = text.str.extract(NUMBER, expand=False)
            numeric = pd.to_numeric(text, errors="coerce")
            result[column] = numeric.fillna(pd.to_numeric(numbers, errors="coerce")).astype("Float64")
    return result


def prepare_echo_records(frame: pd.DataFrame, cutoff: str | None = None) -> pd.DataFrame:
    """Select adult, valid, unique examinations and extract report features."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    rows = frame.copy()
    birth = pd.to_datetime(rows["생년월일"], errors="coerce")
    exam = pd.to_datetime(rows["검사시행일"], errors="coerce")
    # Count completed birthdays instead of subtracting calendar years.
    rows["age_years"] = exam.dt.year - birth.dt.year - (
        (exam.dt.month < birth.dt.month)
        | ((exam.dt.month == birth.dt.month) & (exam.dt.day < birth.dt.day))
    ).astype(int)
    valid = birth.notna() & exam.notna() & (exam >= birth) & (rows["age_years"] >= 19)
    reports = rows["검사결과"].astype("string").str.strip()
    valid &= reports.notna() & reports.ne("").fillna(False) & reports.ne("1").fillna(False)
    valid &= rows["환자번호"].notna()
    if cutoff is not None:
        valid &= exam.dt.normalize() <= pd.Timestamp(cutoff).normalize()

    rows = rows.loc[valid].copy()
    rows["_exam_date"] = exam.loc[valid].dt.normalize()
    rows = rows.sort_values(["환자번호", "_exam_date"], kind="stable")
    rows = rows.drop_duplicates(["환자번호", "_exam_date"], keep="first")
    if rows.empty:
        features = pd.DataFrame(index=rows.index, columns=FEATURE_COLUMNS)
    else:
        features = pd.DataFrame(
            [extract_measurements(report) for report in rows["검사결과"].astype(str)],
            index=rows.index,
        )
    return pd.concat([rows.drop(columns="_exam_date"), features], axis=1).reset_index(drop=True)


def build_intermediate_table(
    records: pd.DataFrame, existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create or populate an echo table from prepared records using patient/date keys.

    An existing table must contain patient ID and exam date. Only matched rows
    and nonmissing extracted measurements are updated; its other columns stay.
    """
    required = {"환자번호", "검사시행일", *FEATURE_COLUMNS}
    if missing := required - set(records.columns):
        raise ValueError(f"Prepared records missing columns: {', '.join(sorted(missing))}")

    source = records.copy()
    source["검사일자"] = pd.to_datetime(source["검사시행일"], errors="coerce").dt.normalize()
    if source["검사일자"].isna().any():
        raise ValueError("Prepared records contain an invalid exam date")
    if source.duplicated(["환자번호", "검사일자"]).any():
        raise ValueError("Prepared records have duplicate patient/exam-date keys")

    if existing is None:
        columns = ["환자번호", "검사일자"]
        for original, output in DEMOGRAPHIC_COLUMNS.items():
            if original in source.columns:
                source[output] = source[original]
                columns.append(output)
        if "age_years" in source:
            columns.append("age_years")
        return standardize_measurement_columns(source[columns + FEATURE_COLUMNS]).reset_index(drop=True)

    if not {"환자번호", "검사일자"}.issubset(existing.columns):
        raise ValueError("Existing table requires 환자번호 and 검사일자 columns")
    result = standardize_measurement_columns(existing).reset_index(drop=True)
    result["_date_key"] = pd.to_datetime(result["검사일자"], errors="coerce").dt.normalize()
    if result["_date_key"].isna().any():
        raise ValueError("Existing table contains an invalid exam date")
    if result.duplicated(["환자번호", "_date_key"]).any():
        raise ValueError("Existing table has duplicate patient/exam-date keys")
    lookup = source.set_index(["환자번호", "검사일자"])
    keys = pd.MultiIndex.from_arrays([result["환자번호"], result["_date_key"]])
    matches = keys.isin(lookup.index)
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="object")
        if matches.any():
            new_values = lookup.reindex(keys[matches])[column].to_numpy()
            result[column] = result[column].astype("object")
            for index, value in zip(result.index[matches], new_values):
                if pd.notna(value):
                    result.at[index, column] = value
    return standardize_measurement_columns(result.drop(columns="_date_key"))


def _read_table(source: str | Path) -> pd.DataFrame:
    """Read an authorized local CSV or Excel table."""
    source = Path(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(source)
    raise ValueError("Input must be a .csv, .xlsx, or .xlsm file")


def process_file(
    source: str | Path, destination: str | Path, cutoff: str | None = None,
    existing_source: str | Path | None = None,
) -> pd.DataFrame:
    """Build a new intermediate echo CSV or populate an existing keyed table."""
    prepared = prepare_echo_records(_read_table(source), cutoff=cutoff)
    existing = _read_table(existing_source) if existing_source is not None else None
    result = build_intermediate_table(prepared, existing=existing)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result
