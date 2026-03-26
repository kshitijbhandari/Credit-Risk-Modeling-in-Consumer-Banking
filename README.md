# Credit Risk Modeling (Basel III Framework)

A machine learning project that implements the three core components of the Basel III credit risk framework using historical Lending Club loan data (2007–2014).

---

## Models

### 1. Probability of Default (PD)
Predicts the likelihood that a borrower will default on a loan.
- **Algorithm:** Logistic Regression with custom p-value computation
- **Feature Selection:** LassoCV (5-fold cross-validation)
- **Evaluation:** ROC-AUC, Confusion Matrix, Classification threshold = 0.75
- **Output:** `pd_model.sav` (generated on run, not tracked in repo)

### 2. Loss Given Default (LGD)
Estimates the percentage of the loan amount lost if a default occurs. Uses a two-stage approach:
- **Stage 1 — Classification:** Logistic Regression to predict whether any loss occurs (threshold = 0.45)
- **Stage 2 — Regression:** Linear Regression to estimate the magnitude of loss
- **Evaluation:** ROC-AUC (Stage 1), MSE + R² (Stage 2)
- **Output:** `lgd_model_stage_1.sav`, `lgd_model_stage_2.sav` (generated on run, not tracked in repo)

### 3. Exposure at Default (EAD)
Estimates the Credit Conversion Factor (CCF) — the proportion of the loan outstanding at the time of default.
- **Algorithm:** Linear Regression with custom p-value computation
- **Target:** `CCF = (funded_amnt − total_rec_prncp) / funded_amnt`
- **Evaluation:** R², MSE, Pearson Correlation
- **Output:** `ead_model.sav` (generated on run, not tracked in repo)

---

## Dataset

**Source:** [Lending Club Loan Data (2007–2014)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — available on Kaggle.

| Split    | Samples |
|----------|---------|
| Train    | 373,028 |
| Test     | 93,257  |

**Class Distribution (PD target):** ~89% good loans, ~11% bad loans

**Required raw file:** `loan_data_2007_2014.csv` (place in project root before running preprocessing)

---

## Project Structure

```
Credit risk/
├── 01_Data_Preprocessing.ipynb          # Data cleaning, encoding, feature engineering
├── 02_Probability_of_Default_Model.ipynb # PD model training and evaluation
├── 03_LGD_and_EAD_Models.ipynb          # LGD (2-stage) and EAD model training
│
├── loan_data_2007_2014.csv      # Raw dataset (not included — download separately)
├── loan_data_inputs_train.csv   # Preprocessed training features (generated)
├── loan_data_inputs_test.csv    # Preprocessed test features (generated)
├── loan_data_targets_train.csv  # Training labels
├── loan_data_targets_test.csv   # Test labels
│
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Python version:** 3.9+

---

## Usage

Run the notebooks **in order**:

1. **`01_Data_Preprocessing.ipynb`** — Cleans and preprocesses `loan_data_2007_2014.csv`, outputs train/test CSV files.
2. **`02_Probability_of_Default_Model.ipynb`** — Trains and saves the PD model.
3. **`03_LGD_and_EAD_Models.ipynb`** — Trains and saves the LGD and EAD models.

---

## Key Technical Details

### Custom Statistical Inference
Both logistic and linear regression models are extended to compute **p-values** for every feature coefficient, enabling statistical significance testing beyond standard sklearn outputs.

- **Logistic Regression:** Uses the Fisher Information Matrix to derive coefficient variances, then computes two-tailed z-test p-values.
- **Linear Regression:** Computes t-statistics and p-values using the t-distribution.

### Feature Selection
LassoCV automatically selects significant features for the PD model by shrinking irrelevant coefficients to zero, reducing noise and improving generalization.

---

## Results

### Probability of Default (PD)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.690 |
| Accuracy (threshold = 0.75) | 87.3% |

**Confusion Matrix (test set, 93,257 samples):**

|  | Predicted Good | Predicted Default |
|--|---------------|------------------|
| **Actual Good** | 1,066 | 9,128 |
| **Actual Default** | 2,680 | 80,383 |

---

### Loss Given Default — Stage 1 (Classification)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.620 |
| Accuracy (threshold = 0.45) | 57.7% |

**Confusion Matrix:**

|  | Predicted 0 | Predicted 1 |
|--|------------|------------|
| **Actual 0** | 523 | 3,250 |
| **Actual 1** | 410 | 4,465 |

---

### Loss Given Default — Stage 2 (Regression)

| Metric | Value |
|--------|-------|
| Pearson Correlation (actual vs predicted) | 0.325 |
| Mean predicted LGD | 8.9% |
| Std | 5.0% |
| Range | 0% – 23.5% |

---

### Exposure at Default (EAD)

| Metric | Value |
|--------|-------|
| R² | 0.278 |
| MSE | 0.0290 |
| Pearson Correlation | 0.527 |
| Mean predicted CCF | 73.6% |

---

## Expected Loss Formula

After training all three models, the expected loss for any loan can be estimated as:

```
Expected Loss (EL) = PD × LGD × EAD
```

This is the standard Basel III formula used by financial institutions to quantify credit risk.
