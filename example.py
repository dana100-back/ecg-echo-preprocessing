"""Run the preprocessing pipeline on invented example records."""

import pandas as pd

from echo_preprocessing import build_intermediate_table, prepare_echo_records


example = pd.DataFrame([
    {"환자번호": "EXAMPLE-001", "생년월일": "1980-06-15", "검사시행일": "2022-07-20",
     "검사결과": "LVEF: 43.5 %\nLVID s: 3.2 cm\nLA size: 4.1 cm\nMV E V: 0.72 m/s\nMV E': 0.08 m/s\nA-fib during examination\nhypokinesia"},
    {"환자번호": "EXAMPLE-002", "생년월일": "2005-07-21", "검사시행일": "2022-07-20",
     "검사결과": "EF: 60 %"},
])

if __name__ == "__main__":
    result = build_intermediate_table(prepare_echo_records(example, cutoff="2022-07-25"))
    print(result[["환자번호", "age_years", "ef", "lvesd", "la_size", "e_prime", "rwma_mentioned", "afib_during_exam_mentioned"]].to_string(index=False))
