import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv(r"C:\Users\saipr\Downloads\archive\Titanic-Dataset.csv")
# Display first five rows
print(df.head())
# Number of rows and columns
print(df.shape)

# Column names
print(df.columns)

# Data types and non-null counts
print(df.info())

# Statistical summary
print(df.describe())

# Summary including categorical columns
print(df.describe(include="all"))
# Count missing values
print(df.isnull().sum())

# Missing value percentages
missing_percent = (df.isnull().sum() / len(df)) * 100

print(missing_percent)

# Fill Age using median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill Embarked using mode
df['Embarked'] = df['Embarked'].fillna(
    df['Embarked'].mode()[0]
)

# Drop Cabin because it has many missing values
df = df.drop(columns=['Cabin'])
# Check duplicates
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print(df.shape)

plt.figure(figsize=(8, 5))

plt.boxplot(df["Fare"])

plt.title("Fare Outlier Detection")
plt.ylabel("Fare")

plt.show()

# Convert categorical columns into numbers
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# One-hot encode Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    dtype=int
)

print(df.head())


# Select numerical columns
columns = ["Age", "Fare"]

# Standardization
for col in columns:
    mean = np.mean(df[col])
    std = np.std(df[col])

    if std != 0:
        df[col] = (df[col] - mean) / std

print(df[columns].head())
plt.figure(figsize=(8, 5))

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

survival = df["Survived"].value_counts()

plt.bar(
    ["Not Survived", "Survived"],
    survival.sort_index()
)

plt.title("Titanic Survival Distribution")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")

plt.show()

# Check missing values
print(df.isnull().sum())

# Check dataset information
print(df.info())

# Save cleaned dataset
df.to_csv("Cleaned_Titanic.csv", index=False)

