# 🏥 Hospital-Acquired Infection (HAI) Prediction Dataset

![License (Code)](https://img.shields.io/badge/License%20(Code)-MIT-yellow.svg)
![License (Data)](https://img.shields.io/badge/License%20(Data)-CC%20BY--NC%204.0-lightgrey.svg)
![Records](https://img.shields.io/badge/Records-119,743-blue.svg)
![Features](https://img.shields.io/badge/Features-32-green.svg)

A comprehensive dataset of inpatient admission records from emergency and hospital wards, including demographic information, vital signs, laboratory results, and clinical outcomes.

---

## 📊 Overview

This dataset includes **119,743 hospital admission records** with **32 clinical and demographic features**. The data is derived from real‑world hospital records spanning the years **June 2023 – February 2026**, and included in a file, named [Hospital_infection_data.csv](Hospital_infection_data.csv).

### General Statistics

- **Total Records**: 119,743
- **Total Features**: 32
- **Time Period**: June 2023 – February 2026
- **Unique Complaints**: 901 types

---

## 📋 Feature Description

### Demographic Information

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `Age`  | Patient's age (years) | Float | 0.01% |
| `Sex`  | Gender (Male/Female) | String | 0% |

### Admission and Ward Information

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `Year` | Admission year (Solar Hijri) | String | 0% |
| `Department` | Hospital ward | String | 0.01% |
| `Triage level` | Triage level (1 to 5) | Integer | 0% |
| `Patient complaint` | Chief complaint | String | 4.1% |
| `diagnosis` | ICD‑10 diagnosis code | String | 63.2% |

### Vital Signs

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `SPo2` | Blood oxygen saturation (%) | Float | 0.63% |
| `BPMin` | Diastolic blood pressure (mmHg) | Float | 2.27% |
| `BPMax` | Systolic blood pressure (mmHg) | Float | 2.27% |
| `PR` | Pulse rate (bpm) | Float | 0.75% |
| `RR` | Respiratory rate (per minute) | Float | 87.45% |
| `T` | Body temperature (°C) | String | 1.51% |

### Blood Tests

| Column | Description | Data Type | Missing % | Unit |
| ------ | ----------- | --------- | --------- | ---- |
| `BS` | Random blood sugar | Float | 74.02% | mg/dL |
| `BS (second)` | Blood sugar (second measurement) | Float | 4.70% | mg/dL |
| `WBC` | White blood cell count | Float | 2.48% | ×10³/µL |
| `RBC` | Red blood cell count | Float | 93.21% | — |
| `HB` | Hemoglobin | Float | 0.59% | g/dL |
| `HCT` | Hematocrit | Float | 0.59% | % |
| `PLT` | Platelet count | Float | 0.66% | ×10³/µL |
| `ESR` | Erythrocyte sedimentation rate | Float | 81.73% | mm/hr |
| `CRP` | C‑reactive protein | String (Qualitative) | 53.82% | — |

### Blood Biochemistry

| Column | Description | Data Type | Missing % | Unit |
| ------ | ----------- | --------- | --------- | ---- |
| `UREA` | Blood urea | Float | 1.26% | mg/dL |
| `CR` | Creatinine | Float | 1.17% | mg/dL |
| `NA` | Sodium | Float | 16.12% | mEq/L |
| `K` | Potassium | Float | 19.21% | mEq/L |

### Coagulation

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `PT` | Prothrombin time | String | 99.25% |

### Target Variable

| Column | Description | Data Type | Missing % |
| ------ | ----------- | --------- | --------- |
| `Label` | Clinical outcome (0 or 1) | Integer | 0% |

---

## ⚠️ Data Quality Notes

### Missing Data

Some columns have significant percentages of missing data:

| Status | Missing % | Columns |
| ------ | --------- | ------- |
| **Critical** | > 80% | RBC (93.2%), PT (99.3%), ESR (81.7%), RR (87.5%) |
| **High** | 50–80% | BS (74%), CRP (53.8%), diagnosis (63.2%) |
| **Medium** | 10–20% | NA (16.1%), K (19.2%) |
| **Good** | < 10% | Other columns |

### Class Imbalance

The target variable (`Label`) is imbalanced:

- **Negative Class (0)**: 98.62% (118,083 cases)
- **Positive Class (1)**: 1.38% (1,660 cases)

---

## 📝 Usage Guide

### Prerequisites

For working with this dataset, the following libraries are recommended:

- Python 3.7+
- pandas (data loading and processing)
- numpy (numerical computations)
- matplotlib/seaborn (visualization)
- scikit‑learn (machine learning modeling)
- hazm or parsivar (Persian text processing)

---

## 🧑‍⚖️ License

This repository contains two distinct components with different licenses:

1. **Source Code (Python scripts)**: Licensed under the **MIT License** – see the [LICENSE](LICENSE) file.
2. **Dataset (CSV files)**: Licensed under the **Creative Commons Attribution‑NonCommercial 4.0 International License (CC BY‑NC 4.0)** – see the [LICENSE_DATA](LICENSE_DATA) file.
   - This license permits **non‑commercial use**, sharing, and adaptation, provided appropriate credit is given.
   - **Commercial use** (including selling, licensing, or using the data in proprietary commercial products) is **strictly prohibited** without explicit written permission from the authors.
   - Users **must not attempt to re‑identify** any patients, although all data has been de-identified.
   - **Additional Terms of Use:**

      By downloading, accessing, or using this dataset, you agree to the following additional terms:

      **Citation Requirement:** You must cite the associated Data Descriptor paper in any publication, presentation, or product that uses or references this dataset:

      > Bakhshi, Z., Ghasemi, V., & Zamanian, M. H. (2026). A Large‑Scale Hospital Dataset for Early Prediction of Hospital‑Acquired Infections Based on Admission and First‑Day Clinical Data. *Journal of Biomedical Physics and     Engineering*. [DOI: To be Inserted]
   
      **Dataset Citation (for data access):**
      > Bakhshi, Z., Ghasemi, V., & Zamanian, M. H. (2026). A Large‑Scale Hospital Dataset for Early Prediction of Hospital‑Acquired Infections Based on Admission and First‑Day Clinical Data (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/[insert Zenodo DOI]
   
      **Attribution:** As required by the CC BY-NC 4.0 license, you must give appropriate credit to the dataset creators and provide a link to the license.
---

## 🛡️ Ethical Approval

This study was conducted in compliance with the **Declaration of Helsinki** and approved by the **Ethics Committee of the School of Medicine, Kermanshah University of Medical Sciences** under approval code:

**IR.KUMS.MED.REC.1404.284**

All patient data were fully anonymized prior to extraction, and the requirement for informed consent was waived by the Ethics Committee due to the retrospective nature of the study.

---

## 👥 Contributors

- **Zahra Bakhshi** – Data collection, preprocessing, and analysis
- **Dr. Vahid Ghasemi** – Supervisor
- **Dr. Mohammadhossein Zamanian** – advisor

---

## 📧 Contact and Collaboration

For questions, suggestions, or collaboration on research projects based on this data:

- ✉️ Email (Corresponding Author): [zahrabakhshi22@gmail.com](mailto:zahrabakhshi22@gmail.com)
- 📧 Email (Supervisor): [v.ghasemi921@gmail.com](mailto:v.ghasemi921@gmail.com)
- 📧 Email (Advisor): [mohammadhossein.zamanian@kums.ac.ir](mailto:mohammadhossein.zamanian@kums.ac.ir)

---
## Validation and Code

We validated the dataset using classical machine learning methods to establish baseline performance for hospital-acquired infection (HAI) prediction on a subset of the dataset. The code is available in the **`CODE/`** subdirectory, in Python.

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

Navigate to the `CODES/` folder and run:

```bash
python CODE.py
```

The script will:
- Preprocess the data
- Train classifiers (Logistic Regression, Random Forest, AdaBoost, XGBoost)
- Evaluate performance using cross-validation
- Generate metrics and visualizations
