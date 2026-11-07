# Data-Cleaning-and-Preprocessing-for-Titanic-Dataset-
Objective: How to clean and prepare raw data for ML. 
Tools: Python, Pandas , Numpy, Matplotlib.
-> Import the dataset and explore basic info (nulls, data types )
-> Handle missing values using mean , median / imputation
-> Convert categorical features into numerical using encoding
-> Normalize/ Standardize the numerical features 
-> Visualize outliers using boxplots and remove them.

## Project Overview

This project demonstrates how to clean and prepare the Titanic passenger
dataset for a machine learning workflow using Python, Pandas, NumPy, and
Matplotlib.

The preprocessing steps include: - Loading and exploring a CSV dataset -
Inspecting data types and summary statistics - Identifying and handling
missing values - Checking and removing duplicate rows - Visualizing
potential outliers - Encoding categorical variables - Standardizing
selected numerical features - Visualizing feature distributions and
survival counts - Exporting the cleaned dataset

## Tools and Libraries

-   Python 3
-   Pandas
-   NumPy
-   Matplotlib

## Project Files

``` text
project-folder/
├── Titanic-Dataset.csv
├── your_python_script.py
├── Cleaned_Titanic.csv
└── README.md
```

Replace `your_python_script.py` with the actual name of your Python
file. `Cleaned_Titanic.csv` is generated when the script runs
successfully.

## Installation

Install the required libraries:

``` bash
python -m pip install pandas numpy matplotlib
```

If `python` is not recognized on Windows, try:

``` bash
py -m pip install pandas numpy matplotlib
```

## Dataset Setup

Place `Titanic-Dataset.csv` in the project folder, or update the CSV
path in your Python code.

Example:

``` python
import pandas as pd
df = pd.read_csv("Titanic-Dataset.csv")
```

On Windows, you can also use a raw absolute path:

``` python
df = pd.read_csv(r"C:\Users\YourName\Downloads\Titanic-Dataset.csv")
```

Change the path to match the actual location of your dataset.

## Preprocessing Workflow

### 1. Load and inspect the data

Use `head()`, `shape`, `info()`, `describe()`, and `columns` to
understand the dataset.

### 2. Find missing values

``` python
print(df.isnull().sum())
```

### 3. Handle missing values

-   Fill `Age` with its median.
-   Fill `Embarked` with its most frequent value (mode).
-   Drop `Cabin` in this introductory workflow because it contains many
    missing entries.

``` python
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
```

### 4. Remove duplicate rows

``` python
df = df.drop_duplicates()
```

### 5. Visualize possible outliers

A box plot of `Fare` can help identify unusually low or high values.
This workflow visualizes outliers; it does not automatically remove or
cap them.

### 6. Encode categorical columns

-   Map `Sex` to numeric values.
-   Apply one-hot encoding to `Embarked`.

### 7. Standardize selected numerical columns


The example standardizes `Age` and `Fare` using the mean and standard
deviation.

**Machine-learning note:** For model evaluation, split the data into
training and test sets before fitting imputation or scaling parameters.
Fit preprocessing only on the training data, then apply the learned
transformations to the test data. This avoids data leakage.

### 8. Visualize and export

The script creates an age histogram and a survival-count bar chart, then
saves the processed data as `Cleaned_Titanic.csv`.

## How to Run

1.  Open the project folder in your Python editor.
2.  Confirm that `Titanic-Dataset.csv` is in the folder or update its
    path.
3.  Install the required libraries.
4.  Run your Python script.
5.  Check the output plots and the generated `Cleaned_Titanic.csv` file.

## Notes and Limitations

-   This is an educational preprocessing example, not a complete
    predictive model.
-   The `Survived` column is the target variable in the Titanic dataset.
-   Do not use the target column as an input feature when training a
    model.
-   Feature selection and preprocessing choices should be validated for
    the intended task.
-   The example standardizes the full dataset for demonstration. Use a
    train/test split and training-only fitting for a leakage-safe ML
    pipeline.

## Author
priyankasrv_18

Add your name here.
