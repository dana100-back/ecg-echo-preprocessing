"""Behavior checks using entirely invented records."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from echo_preprocessing import PATTERNS, build_intermediate_table, extract_measurements, prepare_echo_records, process_file, standardize_measurement_columns


def record(patient, birth, exam, report):
    return {"환자번호": patient, "생년월일": birth, "검사시행일": exam, "검사결과": report}


class PreprocessingTests(unittest.TestCase):
    def test_standardized_names_and_numeric_values_from_legacy_columns(self):
        legacy = pd.DataFrame({
            "EF": ["EF: 55 %"], "LVESD_val": ["32.5"], "LVEDD_val": ["49"],
            "LA size": ["LA size: 4.2 cm"], "LA_volume_val": ["62"], "LAVI_val": ["31.8"],
        })
        result = standardize_measurement_columns(legacy)
        expected = {"ef": 55, "lvesd": 32.5, "lvedd": 49, "la_size": 4.2, "la_volume": 62, "lavi": 31.8}
        for column, value in expected.items():
            self.assertEqual(result.loc[0, column], value)
            self.assertTrue(pd.api.types.is_numeric_dtype(result[column]))
        self.assertFalse(set(legacy.columns) & set(result.columns))
        self.assertEqual(set(result.columns), set(PATTERNS))

    def test_filters_adults_valid_reports_cutoff_and_duplicate_dates(self):
        frame = pd.DataFrame([
            record("a", "2003-07-21", "2022-07-20", "EF: 50"),
            record("B", "2003-07-20", "2022-07-20", "EF: 40"),
            record("B", "2003-07-20", "2022-07-20 13:00", "EF: 30"),
            record("C", "1980-01-01", "2022-07-20", "1"),
            record("D", "1980-01-01", "2022-07-20", "  "),
            record("e", "1980-01-01", "2022-07-26", "EF: 60"),
            record("F", "bad-date", "2022-07-20", "EF: 60"),
        ])
        result = prepare_echo_records(frame, cutoff="2022-07-25")
        self.assertEqual(result["환자번호"].tolist(), ["B"])
        self.assertEqual(result.loc[0, "age_years"], 19)
        self.assertEqual(result.loc[0, "ef"], 40.0)

    def test_extracts_variants_and_separates_doppler_ratio(self):
        report = "LVEF: 43.5 %\nLVID d: 5.1 cm\nLA (M-mode): 4.2 cm\nE/E': 12.0\nMV E V: 0.72\nMV E': 0.08\nA-fib during examination\nHypokinesia"
        result = extract_measurements(report)
        self.assertEqual(result["ef"], 43.5)
        self.assertEqual(result["lvedd"], 5.1)
        self.assertEqual(result["la_size"], 4.2)
        self.assertEqual(result["e"], 0.72)
        self.assertEqual(result["e_prime"], 0.08)
        self.assertTrue(result["rwma_mentioned"])
        self.assertTrue(result["afib_during_exam_mentioned"])

    def test_missing_columns_and_csv_roundtrip(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            prepare_echo_records(pd.DataFrame({"환자번호": ["a"]}))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            destination = Path(directory) / "out" / "processed.csv"
            pd.DataFrame([record("X", "1980-01-01", "2022-07-20", "EF: 55")]).to_csv(source, index=False)
            result = process_file(source, destination)
            self.assertEqual(result.loc[0, "ef"], 55.0)
            self.assertEqual(pd.read_csv(destination).loc[0, "ef"], 55.0)
            self.assertNotIn("검사결과", result.columns)

    def test_no_matching_records_retains_output_schema(self):
        frame = pd.DataFrame([record("Y", "2020-01-01", "2022-07-20", "EF: 55")])
        result = prepare_echo_records(frame)
        self.assertTrue(result.empty)
        self.assertIn("ef", result.columns)

    def test_builds_intermediate_table_and_matches_existing_rows_by_key(self):
        raw = pd.DataFrame([
            {**record("B", "1980-01-01", "2022-07-20", "EF: 44.5\nLA vol: 62\nMV A': 0.09"),
             "성별": "F", "신장(cm)": 160, "체중(kg)": 53, "BMI(kg/m²)": 20.7},
            record("a", "1980-01-01", "2022-07-20", "EF: 55"),
        ])
        prepared = prepare_echo_records(raw)
        table = build_intermediate_table(prepared)
        self.assertNotIn("검사결과", table.columns)
        self.assertEqual(table.loc[table["환자번호"].eq("B"), "ef"].iloc[0], 44.5)
        self.assertEqual(table.loc[table["환자번호"].eq("B"), "la_volume"].iloc[0], 62)
        self.assertEqual(table.loc[table["환자번호"].eq("B"), "a_prime"].iloc[0], 0.09)
        self.assertEqual(table.loc[table["환자번호"].eq("B"), "신장"].iloc[0], 160)

        existing = pd.DataFrame([
            {"환자번호": "B", "검사일자": "2022-07-20", "EF": 1, "note": "keep B"},
            {"환자번호": "C", "검사일자": "2022-07-20", "EF": 9, "note": "keep C"},
            {"환자번호": "a", "검사일자": "2022-07-20", "EF": 2, "note": "keep A"},
        ])
        merged = build_intermediate_table(prepared, existing)
        self.assertEqual(merged["환자번호"].tolist(), ["B", "C", "a"])
        self.assertEqual(merged["ef"].tolist(), [44.5, 9, 55])
        self.assertNotIn("EF", merged.columns)
        self.assertEqual(merged["note"].tolist(), ["keep B", "keep C", "keep A"])


if __name__ == "__main__":
    unittest.main()
