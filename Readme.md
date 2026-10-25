# 🏥 Hospital-Acquired Infection (HAI) Prediction Dataset

![License (Code)](https://img.shields.io/badge/License%20(Code)-MIT-yellow.svg)
![License (Data)](https://img.shields.io/badge/License%20(Data)-CC%20BY--NC%204.0-lightgrey.svg)
![Records](https://img.shields.io/badge/Records-119,743-blue.svg)
![Predictors](https://img.shields.io/badge/Predictors-26-green.svg)
![Columns](https://img.shields.io/badge/Columns-27-blue.svg)

A comprehensive dataset of in-patient admission records from emergency and hospital wards, including demographic information, triage vital signs, first-24-hour laboratory results, and the HAI outcome label.

---

## 📄 Overview

This dataset includes **119,743 hospital admission records** with **27 data columns** excluding the row index (`#`): **26 predictors + 1 target label**. The data are derived from real-world hospital records spanning **22 June 2023 – 19 February 2026** (1 Tir 1402 – 30 Bahman 1404, Persian calendar) at **Imam-Reza Hospital, Kermanshah, Iran**. The de-identified dataset is provided in:

`Hospital_infection_data.csv`

The first column, `#`, is only a row index and should **not** be used as a predictor.

### General Statistics

- **Total Records**: 119,743
- **Data Columns (excluding `#`)**: 27
- **Predictors**: 26
- **Target Variable**: 1 (`Label`)
- **Time Period**: 22 June 2023 – 19 February 2026 (1 Tir 1402 – 30 Bahman 1404, Persian calendar)
- **Unique Patient Complaints**: 900
- **Unique Wards**: 41
- **Unique ICD-10 Diagnoses**: 1,825
- **Triage Levels**: 1–5
- **Gender Categories**: Male / Female
- **CRP Categories**: 7 levels
- **CSV Encoding**: UTF-8-SIG

---

## 📋 Feature Description

### Demographic and Administrative Information

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `Age` | Patient age in years | Numeric | 0.00% |
| `Ward` | Hospital ward of admission | Categorical | 0.00% |
| `Gender` | Gender (Male/Female) | Categorical | 0.00% |
| `Triage_Level` | Triage level (1–5) | Numeric | 0.00% |


### Triage and Clinical Presentation

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `Patient complaint` | Presenting complaint / chief complaint | Categorical / free text | 4.55% |
| `Diagnosis` | Primary diagnosis, ICD-10 code | Categorical | 63.16% |

### Vital Signs at Triage

| Column | Description | Data Type | Missing % | Unit |
| ------ | ----------- | --------- | --------- | ---- |
| `SPo2` | Oxygen saturation | Numeric | 0.89% | % |
| `BS_Triage` | Blood sugar at triage | Numeric | 73.99% | mg/dL |
| `BPMin` | Diastolic blood pressure | Numeric | 2.18% | mmHg |
| `BPMax` | Systolic blood pressure | Numeric | 2.18% | mmHg |
| `T` | Body temperature | Numeric | 1.50% | °C |
| `PR` | Pulse rate | Numeric | 0.74% | bpm |
| `RR` | Respiratory rate | Numeric | 86.94% | breaths/min |

### First-24-Hour Laboratory Results

| Column | Description | Data Type | Missing % | Unit |
| ------ | ----------- | --------- | --------- | ---- |
| `WBC` | White blood cell count | Numeric | 2.39% | ×10³/μL |
| `RBC` | Red blood cell count | Numeric | 2.39% | ×10⁶/μL |
| `HB` | Hemoglobin | Numeric | 0.56% | g/dL |
| `HCT` | Hematocrit | Numeric | 0.57% | % |
| `PLT` | Platelet count | Numeric | 0.64% | ×10³/μL |
| `ESR` | Erythrocyte sedimentation rate | Numeric | 81.69% | mm/hr |
| `CRP` | C-reactive protein | Categorical, 7 levels | 53.74% | — |
| `UREA` | Blood urea | Numeric | 1.20% | mg/dL |
| `CR` | Creatinine | Numeric | 1.10% | mg/dL |
| `NA` | Sodium | Numeric | 16.06% | mEq/L |
| `K` | Potassium | Numeric | 19.16% | mEq/L |
| `PT` | Prothrombin time | Numeric | 99.25% | seconds |
| `BS_Lab` | Laboratory blood sugar | Numeric | 4.64% | mg/dL |

### Target Variable

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `Label` | HAI outcome: `0 = non-HAI`, `1 = HAI` | Integer / Binary | 0.00% |

---

## ⚠️ Data Quality Notes

### Missing Data

Some columns have substantial missingness. This reflects indication-based ordering in routine clinical practice, not necessarily data-entry failure.

| Status | Missing % | Columns |
| ------ | --------- | ------- |
| **Critical** | > 80% | `PT` (99.25%), `RR` (86.94%), `ESR` (81.69%) |
| **High** | 50–80% | `BS_Triage` (73.99%), `Diagnosis` (63.16%), `CRP` (53.74%) |
| **Medium** | 10–20% | `K` (19.16%), `NA` (16.06%) |
| **Good** | < 10% | Other columns |

Overall per-record missingness:

- **Mean**: 19.24%
- **Median**: 18.52%
- **Minimum**: 0.00%
- **Maximum**: 62.96%
- **Records with >70% missing**: 0

### Class Imbalance

The target variable (`Label`) is imbalanced:

- **Negative Class (0)**: 118,083 cases — **98.614%**
- **Positive Class (1)**: 1,660 cases — **1.386%**
- **Imbalance ratio**: approximately **1 : 71.13**

---

## 📝 Dataset Usage, Validation and Code

We validated the dataset using classical machine-learning methods to establish baseline performance for hospital-acquired infection (HAI) prediction on a subset of the dataset. The code is available in the **`CODE/`** subdirectory in Python.

The final feature set used in the illustrative baseline experiment contains 23 predictors after excluding administrative and NLP-required variables (`Ward`, `Patient complaint`, and `Diagnosis`). This is only a use-case experiment and is not part of the dataset validity evidence.

### Requirements

The scripts require the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

Install them with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Running the Code

Navigate to the `CODE/` folder and run:

```bash
python CODE.py
```

The script will:

- Preprocess the data
- Train classifiers: Logistic Regression, Random Forest, AdaBoost, and XGBoost
- Evaluate performance using cross-validation
- Generate metrics and visualizations

---

## 🧑‍⚖️ License

This repository contains two distinct components with different licenses:

1. **Source Code (Python scripts)**: Licensed under the **MIT License** — see the `LICENSE` file.
2. **Dataset (CSV files)**: Licensed under the **Creative Commons Attribution–NonCommercial 4.0 International License (CC BY–NC 4.0)** — see the `LICENSE_DATA` file.
   - This license permits **non-commercial use**, sharing, and adaptation, provided appropriate credit is given.
   - **Commercial use** is **strictly prohibited** without explicit written permission from the authors.
   - Users **must not attempt to re-identify** any patients, although all data have been de-identified.

   **Additional Terms of Use:**

   By downloading, accessing, or using this dataset, you agree to the following additional terms:

   **Citation Requirement:** You must cite the associated Data Descriptor paper in any publication, presentation, or product that uses or references this dataset:

   > Bakhshi, Z., Ghasemi, V., & Zamanian, M. H. (2026). A Large-Scale Hospital Dataset for Early Prediction of Hospital-Acquired Infections. *Submitted to The Journal of Biomedical Physics and Engineering*. [DOI: To be Inserted]

   **Dataset Citation (for data access):**

   > Bakhshi, Z., Ghasemi, V., & Zamanian, M. H. (2026). A Large-Scale Hospital Dataset for Early Prediction of Hospital-Acquired Infections (Version 1.0.0) [Dataset]. Available online: https://doi.org/10.5281/zenodo.21777046

   **Attribution:** As required by the CC BY-NC 4.0 license, you must give appropriate credit to the dataset creators and provide a link to the license.

---

## 🛡️ Ethical Approval

This study was conducted in compliance with the **Declaration of Helsinki** and approved by the **Ethics Committee of the School of Medicine, Kermanshah University of Medical Sciences** under approval code: **IR.KUMS.MED.REC.1404.284**

All patient data were fully anonymized prior to extraction, and the requirement for informed consent was waived by the Ethics Committee due to the retrospective nature of the study.

---

## 👥 Contributors

- **Zahra Bakhshi, MSc** — Department of Computer Engineering, Faculty of Information Technology, Kermanshah University of Technology, Kermanshah, Iran.
- **Dr. Vahid Ghasemi, PhD** — Department of Computer Engineering, Faculty of Information Technology, Kermanshah University of Technology, Kermanshah, Iran.
- **Dr. Mohammad Hossein Zamanian, PhD** — Clinical Research Development Center, Imam-Reza Hospital, Kermanshah University of Medical Sciences, Kermanshah, Iran.

---

## 📧 Contact and Collaboration

For questions, suggestions, or collaboration on research projects based on these data:

- ✉️ Email (Corresponding Author): [v.ghasemi921@gmail.com](mailto:v.ghasemi921@gmail.com)
- 📧 Email (First Author): [zahrabakhshi22@gmail.com](mailto:zahrabakhshi22@gmail.com)
- 📧 Email (Advisor): [mohammadhossein.zamanian@kums.ac.ir](mailto:mohammadhossein.zamanian@kums.ac.ir)
