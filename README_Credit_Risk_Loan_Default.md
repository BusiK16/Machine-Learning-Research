# Credit Risk & Loan Default Prediction

## Project Overview

This project applies supervised machine learning classification to
predict whether a borrower will default on a loan. It covers data
inspection, exploratory data analysis, preprocessing, model training,
test-set evaluation and cross-validation.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn and
Databricks.

## Research Question

**Can machine learning classification models use borrower and loan
information to predict whether a loan will default?**

The target column is `loan_status`: - `0` = No loan default - `1` = Loan
default

This is an educational project. Predictions should not be used on their
own to approve or reject real loan applications.

## Dataset

The working dataset contains **32,416 records**, with 11 predictor
features and one target variable.

  Feature                        Description
  ------------------------------ ---------------------------------------
  `person_age`                   Borrower's age
  `person_income`                Borrower's income
  `person_home_ownership`        Home ownership category
  `person_emp_length`            Employment length
  `loan_intent`                  Purpose of the loan
  `loan_grade`                   Loan grade
  `loan_amnt`                    Loan amount
  `loan_int_rate`                Loan interest rate
  `loan_percent_income`          Loan amount as a proportion of income
  `cb_person_cred_hist_length`   Length of credit history
  `loan_status`                  Target: 0 = no default; 1 = default

### Target distribution

  Loan status   Meaning        Records   Percentage
  ------------- ------------ --------- ------------
  0             No default      24,715       78.41%
  1             Default          6,807       21.59%

The target is imbalanced because no-default records are more common than
default records.

**Dataset sharing note:** Retain the original dataset source and licence
details. Do not publish personal or confidential data without
permission.

## Project Workflow

1.  Import Python libraries.
2.  Load the dataset into a pandas DataFrame.
3.  Inspect the dataset's shape, data types, summary statistics and
    missing values.
4.  Perform exploratory data analysis, including target distribution,
    numerical distributions, categorical values and outlier checks.
5.  Separate predictors (`X`) from the target (`y`) to avoid target
    leakage.
6.  Split the data into training and testing sets using an 80/20
    stratified split.
7.  Preprocess numerical features using median imputation and standard
    scaling.
8.  Preprocess categorical features using most-frequent imputation and
    one-hot encoding.
9.  Fit preprocessing on training data and apply the learned
    transformations to test data.
10. Establish a `DummyClassifier` baseline.
11. Train Logistic Regression, Decision Tree and Random Forest
    classifiers.
12. Evaluate accuracy, precision, recall, F1-score and confusion
    matrices on the test set.
13. Use stratified five-fold cross-validation on the training data.
14. Compare cross-validation and test-set results and discuss
    limitations.

## Models Compared

-   **Logistic Regression:** estimates the probability that a record
    belongs to a class.
-   **Decision Tree:** makes predictions through a sequence of
    feature-based decision rules.
-   **Random Forest:** combines predictions from multiple decision
    trees.
-   **Dummy Classifier:** provides a simple baseline for comparison.

## Results

### Held-out test-set results

  Model                   Accuracy   Precision   Recall   F1-score
  --------------------- ---------- ----------- -------- ----------
  Logistic Regression       0.8690      0.7713   0.5595     0.6485
  Decision Tree             0.8874      0.7286   0.7628     0.7453
  Random Forest             0.9370      0.9763   0.7261     0.8328

### Cross-validation results

  Model                   Accuracy   Precision   Recall   F1-score
  --------------------- ---------- ----------- -------- ----------
  Logistic Regression       0.8666      0.7659   0.5510     0.6407
  Decision Tree             0.8871      0.7275   0.7627     0.7446
  Random Forest             0.9331      0.9642   0.7166     0.8221

### Interpretation

Random Forest achieved the highest accuracy and F1-score in both
comparisons. Its test-set precision was also high. However, its recall
was lower than the Decision Tree's recall, meaning it did not identify
every actual default case.

Recall matters because a false negative is a borrower who actually
defaults but is predicted as no-default. Precision also matters because
false positives may incorrectly flag borrowers who would not default.
Model selection should consider both error types, not accuracy alone.
Cross-validation provides evidence of performance across training folds,
while the held-out test set evaluates performance on data not used to
fit the models.

## Technologies

-   Python
-   Databricks
-   pandas and NumPy
-   Matplotlib and Seaborn
-   scikit-learn

## Suggested Repository Structure

``` text
credit-risk-loan-default/
├── README.md
├── notebooks/
│   └── credit_risk_loan_default.ipynb
├── data/
│   └── README.md
└── reports/
    └── research_report.pdf
```

These names are suggestions. Adjust the structure to match the files
actually in your repository. If the dataset cannot be redistributed, do
not upload it; provide instructions for obtaining it instead.

## How to Run

1.  Clone or download the repository.
2.  Open the notebook in Databricks or a compatible Jupyter environment.
3.  Make the dataset available to the notebook.
4.  Update the file path in the data-loading cell if necessary.
5.  Install missing packages.
6.  Run the notebook from top to bottom.

Example package installation:

``` python
%pip install pandas numpy matplotlib seaborn scikit-learn
```

## Limitations

-   The model learns only from the records and features available in
    this dataset.
-   Results may not generalise to other lenders, populations or economic
    conditions.
-   The dataset contains fewer default cases than no-default cases.
-   The available features may not capture every factor that influences
    loan default.
-   High performance does not prove that predictions are fair,
    explainable or suitable for real lending decisions.
-   Results depend on the train-test split and modelling choices.

## Ethical Considerations

Credit risk models can affect people's access to financial services.
Models may reproduce bias in historical data or make inaccurate
predictions for some groups. Real-world use would require privacy
safeguards, fairness testing, explainability, security and human
oversight. In South Africa, organisations processing personal
information must consider their obligations under the Protection of
Personal Information Act (POPIA).

This project is an educational exercise and is not an automated lending
decision system.

## Project Information

**Assignment:** BrightLearn Machine Learning Research Assignment\
**Project:** Credit Risk and Loan Default Prediction\
**Notebook environment:** Databricks

------------------------------------------------------------------------

*The metrics above reflect the notebook outputs captured during this
project. If you rerun the notebook and results change, update the tables
and interpretation accordingly.*
