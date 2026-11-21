# Titanic Survival Prediction

A practical machine learning project that explores the Titanic passenger dataset, investigates survival patterns, and builds classification models to predict whether a passenger survived the disaster.

The project is implemented in a Jupyter Notebook and follows a structured workflow covering data inspection, exploratory data analysis, preprocessing, model training, and evaluation.

## Project Overview

The objective is to predict the binary target variable `Survived` using passenger information such as age, sex, passenger class, family relationships, ticket fare, and port of embarkation.

The notebook compares three classification approaches:

- **Dummy Classifier** — a simple baseline that always predicts the most frequent class.
- **Logistic Regression** — a linear classification model used as an interpretable benchmark.
- **Random Forest** — an ensemble learning method that captures non-linear relationships between features.

Model performance is assessed using a held-out test set and stratified cross-validation.

## Dataset

The project uses the Titanic passenger dataset containing **891 records and 12 columns**.

The dataset is loaded from the following public CSV source:

`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`

If the remote dataset cannot be accessed, the notebook attempts to load a local file named `train.csv`.

### Target Variable

| Column | Description |
|---|---|
| `Survived` | Survival status: `0` = Did not survive, `1` = Survived |

### Main Features

| Feature | Description |
|---|---|
| `Pclass` | Passenger ticket class (1st, 2nd, or 3rd) |
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `SibSp` | Number of siblings or spouses aboard |
| `Parch` | Number of parents or children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation |
| `Name`, `Ticket`, `Cabin`, `PassengerId` | Additional passenger identifiers and descriptive fields |

## Project Workflow

### 1. Data Loading and Initial Inspection

- Load the dataset using Pandas.
- Inspect the dataset dimensions, data types, and sample records.
- Generate descriptive statistics to understand numerical feature distributions.

### 2. Data Quality Assessment

- Identify missing values and calculate their percentages.
- Check for duplicated records.
- Verify the uniqueness of passenger identifiers.

The initial inspection identifies missing values in `Age`, `Cabin`, and `Embarked`.

### 3. Exploratory Data Analysis (EDA)

The notebook investigates survival patterns across important passenger attributes.

Analysis includes:

- Overall survival class distribution.
- Survival rates by sex.
- Survival rates by passenger class.
- Survival rates by embarkation port.
- Age distributions grouped by survival outcome.
- A heatmap showing survival rates across sex and passenger class, alongside passenger counts.

Matplotlib and Seaborn are used to create visual summaries and support interpretation of the data.

### 4. Train-Test Split

The data is divided into training and testing subsets:

- **Training set:** 80%
- **Testing set:** 20%

The split uses `stratify=y` to preserve the target class distribution in both subsets. A fixed random state (`42`) ensures reproducibility.

### 5. Data Preprocessing and Feature Engineering

A dedicated preprocessing function prepares the input features for machine learning.

The implemented steps include:

- **Missing-value imputation:** Replace missing `Age` values with the training-set median and missing `Embarked` values with the training-set mode.
- **Family size:** Create `FamilySize` using `SibSp + Parch + 1`.
- **Traveling alone:** Create `IsAlone`, indicating whether a passenger has a family size of one.
- **Sex encoding:** Map `male` to `0` and `female` to `1`.
- **Categorical encoding:** Convert `Embarked` into dummy variables using one-hot encoding, with the first category dropped.
- **Feature removal:** Drop `PassengerId`, `Name`, `Ticket`, and `Cabin` from the model input.

The imputation statistics are calculated from the training data and then applied to both the training and test sets to avoid using test-set information during preprocessing.

### 6. Baseline Model

A `DummyClassifier` with the `most_frequent` strategy is used as a reference point.

This baseline provides a simple benchmark that more meaningful machine learning models should aim to outperform.

### 7. Model Training

Two machine learning models are trained and evaluated:

**Logistic Regression**
- Maximum iterations: `1000`
- Random state: `42`
- Evaluated using test-set predictions, a confusion matrix, and a classification report.
- Model coefficients are exponentiated to calculate odds ratios, supporting interpretation of feature relationships.

**Random Forest**
- Number of estimators: `300`
- Random state: `42`
- Evaluated on the held-out test set.
- Feature importance scores are extracted and visualized to show the relative contribution of input features to the model.

### 8. Model Evaluation

The project uses the following evaluation metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Proportion of predicted survivors who actually survived |
| Recall | Proportion of actual survivors correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each class |
| Cross-validation mean | Average accuracy across validation folds |
| Cross-validation standard deviation | Variation in accuracy across folds |

### 9. Stratified Cross-Validation

A five-fold `StratifiedKFold` strategy is used to assess model performance on the training data.

The folds preserve the target class proportions. The notebook reports the mean and standard deviation of accuracy for each model.

A consolidated results table combines test-set metrics with cross-validation statistics and sorts the models by cross-validation mean accuracy.

## Technologies and Libraries

- **Python** — core programming language
- **NumPy** — numerical operations
- **Pandas** — data loading, manipulation, and analysis
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing utilities, classification models, cross-validation, and evaluation metrics
- **Jupyter Notebook** — interactive development environment

## Repository Structure

```text
titanic-survival-prediction/
│
├── Lab1_Titanic_Code.ipynb   # Main notebook containing the complete workflow
├── train.csv                 # Optional local dataset fallback
└── README.md                 # Project documentation
```

`train.csv` is only required when the notebook cannot retrieve the dataset from the remote URL.

## Getting Started

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- Internet access to retrieve the dataset, unless a local `train.csv` file is available

### Installation

1. Clone or download this repository.

2. Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `Lab1_Titanic_Code.ipynb`.

5. Run the notebook cells sequentially from top to bottom.

### Running in Google Colab

Alternatively, upload `Lab1_Titanic_Code.ipynb` to Google Colab and execute the notebook cells in order. Ensure that `train.csv` is uploaded to the Colab working directory if the remote dataset is unavailable.

## Reproducibility

The project sets `RANDOM_STATE = 42` for the train-test split, Logistic Regression, Random Forest, and stratified cross-validation. This supports repeatable results under the same software environment and data source.

## Notes and Limitations

- The notebook is an educational machine learning exercise, not a production deployment.
- The analysis identifies statistical patterns and predictive relationships; it does not establish causal explanations for survival.
- The current preprocessing approach calculates imputation values once from the full training split before cross-validation. For a stricter cross-validation estimate, preprocessing should be placed inside a Scikit-learn `Pipeline` so that imputation and encoding are fitted independently within each fold.
- Model performance may vary if the dataset source or library versions change.

## Learning Outcomes

By completing this project, readers can practice:

- Inspecting and validating a real-world dataset.
- Performing exploratory data analysis and visualizing patterns.
- Handling missing values and encoding categorical variables.
- Creating meaningful features from existing columns.
- Establishing a baseline for classification.
- Training and comparing different classification algorithms.
- Evaluating models with multiple metrics and cross-validation.
- Interpreting model outputs through odds ratios and feature importance.

## License and Dataset Attribution

The notebook uses a publicly available Titanic dataset hosted in the `datasciencedojo/datasets` GitHub repository. Please refer to the dataset source and its applicable terms when redistributing the data.

---

**Project:** Titanic Survival Prediction  
**Task:** Binary Classification  
**Primary Notebook:** `Lab1_Titanic_Code.ipynb`
