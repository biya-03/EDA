# Basic EDA Template

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: pretty plots
sns.set(style="whitegrid")

# 2. Load dataset
df = pd.read_csv("your_dataset.csv")   # or pd.read_excel("file.xlsx")

# 3. Quick overview
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Stats:\n", df.describe(include="all"))

# 4. Duplicate check
print("Duplicates:", df.duplicated().sum())

# 5. Correlation matrix (numeric features)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 6. Univariate analysis
for col in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 7. Categorical features distribution
for col in df.select_dtypes(include="object").columns:
    plt.figure()
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Category Counts for {col}")
    plt.show()

# 8. Outlier detection (boxplots for numeric variables)
for col in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# 9. Pairwise relationships (only if dataset is small)
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()

# 10. Missing value heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()
