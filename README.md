# Breast Cancer Data Preprocessing Pipeline

Multi-pipeline preprocessing system for a breast cancer detection dataset, built as a Data Engineering exam project. The goal is to produce a clean, analysis-ready dataset for downstream machine learning models.

---

## Dataset

| Property | Value |
|---|---|
| Source | [breast_cancer_dataset.csv](https://raw.githubusercontent.com/McNamara10/dataset/refs/heads/main/breast_cancer_dataset.csv) |
| Samples | 569 |
| Features | 30 numeric + 1 categorical (`area error`) + 1 target |
| Target | Binary (0 = benign, 1 = malignant) |
| Missing values | Yes — distributed across multiple features |

---

## Architecture

Three independent preprocessing pipelines are combined via `FeatureUnion`, producing a final `(569, 43)` numpy array.

```
Dataset (569 × 31)
       │
       ├──── Pipeline 1 ──── target=1 records only ──── 30 features
       │       ├── Symmetric numeric   → SimpleImputer(mean)   → StandardScaler
       │       ├── Asymmetric numeric  → SimpleImputer(median) → SkewnessCorrector → StandardScaler
       │       └── Categorical         → SimpleImputer(most_frequent) → OneHotEncoder
       │
       ├──── Pipeline 2 ──── all records ──── 5 features
       │       ├── Numeric   → SimpleImputer(mean)         → KBinsDiscretizer(quantile)
       │       └── Categorical → SimpleImputer(most_frequent) → OrdinalEncoder([A,B,C])
       │                └── SelectKBest(f_classif, k=5)
       │
       └──── Pipeline 3 ──── all records, numeric only ──── 8 features
               └── SimpleImputer(mean) → SkewnessCorrector → StandardScaler → PCA(80%) → MinMaxScaler

                                        FeatureUnion
                                             │
                                    Output: (569, 43)
```

---

## Output Feature Breakdown

| Columns | Pipeline | Description |
|---|---|---|
| 0 – 29 | Pipeline 1 | Preprocessed features for malignant cases (target=1); zeros for benign cases |
| 30 – 34 | Pipeline 2 | Top 5 most informative features via SelectKBest (f_classif) |
| 35 – 42 | Pipeline 3 | 8 principal components explaining 80% of variance |

**Selected features (Pipeline 2):** `mean perimeter`, `worst perimeter`, `worst area`, `worst concavity`, `worst concave points`

---

## Custom Components

### `SkewnessCorrector`
Identifies asymmetric columns (skewness > 0.75) during `fit` and applies `log1p` transformation during `transform`. Compatible with scikit-learn pipelines via `BaseEstimator` and `TransformerMixin`.

### `PipelineWithRowFilter`
Wrapper that applies an inner pipeline only to rows matching a condition function. Used to train and transform exclusively on malignant cases (target=1), filling non-matching rows with zeros.

---

## Project Structure

```
breast-cancer-preprocessing-pipeline/
│
├── project_pre_processing_rilevazione_tumore_al_seno.ipynb   # Main notebook
├── requirements.txt
└── README.md
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/McNamara10/breast-cancer-preprocessing-pipeline.git
cd breast-cancer-preprocessing-pipeline
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
jupyter notebook project_pre_processing_rilevazione_tumore_al_seno.ipynb
```

The dataset is loaded automatically from a remote URL — no local file needed.

---

## Tech Stack

- **Python 3**
- **scikit-learn** — Pipeline, ColumnTransformer, FeatureUnion, transformers, PCA, SelectKBest
- **pandas** — data manipulation and EDA
- **numpy** — numerical operations
- **matplotlib / seaborn** — distribution analysis and visualization

---

## Author

**Marcello Orru** — [GitHub](https://github.com/McNamara10)
