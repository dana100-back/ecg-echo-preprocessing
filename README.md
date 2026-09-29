# ECG–Echocardiogram Data Preparation

An adapted, reproducible portfolio example of echocardiogram preprocessing I worked on during a research internship at Seoul National University Hospital. The original exploratory notebooks filtered examination records and extracted measurements from free-text echo reports for a study of reduced LVEF prediction from atrial fibrillation/flutter ECGs.

## My contribution

- Filtered adult examinations, excluded empty or invalid reports, and removed duplicate patient/examination-date pairs.
- Extracted ejection fraction (EF), left ventricular dimensions, left atrial measurements, Doppler measurements, and wall-motion mentions from report text.
- Checked report mentions of atrial fibrillation during the examination and prepared intermediate tables for subsequent research work.

The `echo_preprocessing.py` module selects eligible reports, extracts features, and creates or populates an intermediate echo table. Its keyword rules are an illustrative adaptation of the original research workflow. **It is not a validated clinical parser or an exact reproduction of the research dataset.** The included example contains invented records and does not reproduce the study's results.

## Run the example

Requires Python 3.10+ and pandas 2.x. `openpyxl` is needed when reading `.xlsx` files.

```bash
python -m pip install -r requirements.txt
python example.py
python -m unittest test_preprocessing -v
```

To process your own authorized table, import `prepare_echo_records` and pass a pandas DataFrame with these source columns: `환자번호` (patient ID), `생년월일` (birth date), `검사시행일` (exam date), and `검사결과` (report). Dates must be parseable by pandas. The optional `cutoff` argument includes examinations on or before that date. `build_intermediate_table` creates a separate echo measurement table. `process_file` runs both steps and writes a CSV.

```python
from echo_preprocessing import process_file

process_file("data/input.xlsx", "data/processed.csv", cutoff="2022-07-25")
```

To update an existing intermediate table, pass `existing_source="data/echo_table.csv"`. It must have `환자번호` and `검사일자` columns. Records are matched by patient ID and normalized examination date, rather than row number. Unmatched rows and unrelated columns are retained; a missing extracted value does not erase an existing measurement. The new table includes patient ID, exam date, available demographic fields, `age_years`, `ef`, `lvesd`, `lvedd`, `la_size`, `la_volume`, `lavi`, `e`, `a`, `s`, `e_prime`, `a_prime`, `s_prime`, `dt`, `rwma_mentioned`, and `afib_during_exam_mentioned`. It omits raw report text. Numeric measurements use the source report's units; unit normalization and medical interpretation are outside this example.

## Limits and data access

The source clinical records and intermediate tables are confidential and are not in this repository. The example only shows that the software works on synthetic cases; it cannot verify accuracy against the hospital dataset. Real report templates vary, and the regexes should be reviewed against authorized data before research reuse. No patient data, notebook output, or machine-specific path is published.

## Origin of this adaptation

The original `ray_echo_demo.ipynb` and `ray_echo_fin.ipynb` were exploratory working notebooks. They contained out-of-order cells, undefined intermediate variables, and conditions such as `('LVIDs' or 'LVID s') in word` that only check the first phrase. This repository presents the same preprocessing stages with explicit inputs, English comments, and executable tests. It does not reproduce the original files' intermediate table schema or their exact cohort and matching rules. The work documents a data preparation contribution, not model training or authorship of the whole study.

## Standardized output names

The final measurement columns use lowercase snake_case and numeric dtypes. Temporary `_val` names and legacy uppercase names are normalized before writing a new table or populating an existing one. Decimal values are preserved. Run `example.py` for an executable synthetic demonstration.

| Legacy column | Final numeric column |
| --- | --- |
| EF | ef |
| LVESD / LVESD_val | lvesd |
| LVEDD / LVEDD_val | lvedd |
| LA size / LA_size_val | la_size |
| LA volume / LA_volume_val | la_volume |
| LAVI / LAVI_val | lavi |
