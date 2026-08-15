# ==========================================================
# MACHINE LEARNING ON ABSENTEEISM AT WORK DATASET
# ==========================================================

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

file_path = r"C:\Users\hp\Desktop\Absenteeism_at_work.csv"

df = pd.read_csv(
    file_path,
    sep=";"
)

print("\n==============================")
print("DATASET")
print("==============================")

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================================
# 3. CHECK DATA
# ==========================================================

print("\n==============================")
print("DATA INFORMATION")
print("==============================")

print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================================
# 4. REMOVE DUPLICATES
# ==========================================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ==========================================================
# 5. CORRELATION ANALYSIS
# ==========================================================

print("\n==============================")
print("CORRELATION ANALYSIS")
print("==============================")

correlation = df.corr(numeric_only=True)

print("\nCorrelation with Absenteeism:")
print(
    correlation["Absenteeism time in hours"]
    .sort_values(ascending=False)
)


# ==========================================================
# 6. CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==========================================================
# 7. FEATURE ENGINEERING
# ==========================================================

print("\n==============================")
print("FEATURE ENGINEERING")
print("==============================")


# New feature 1:
# Workload multiplied by Weight

if "Work load Average/day" in df.columns and "Weight" in df.columns:

    df["Workload_Weight"] = (
        df["Work load Average/day"] *
        df["Weight"]
    )


# New feature 2:
# Create age groups

if "Age" in df.columns:

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 100],
        labels=[1, 2, 3, 4]
    )

    df["Age_Group"] = df["Age_Group"].astype(int)


print("\nNew Features Created:")

print(df.head())


# ==========================================================
# 8. LINEAR REGRESSION
# ==========================================================

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")


# Features

X = df.drop(
    "Absenteeism time in hours",
    axis=1
)

# Target

y = df["Absenteeism time in hours"]


# Keep numerical columns only

X = X.select_dtypes(
    include=["int64", "float64"]
)


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Create Linear Regression model

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)


# Prediction

linear_prediction = linear_model.predict(
    X_test
)


# Evaluation

mae = mean_absolute_error(
    y_test,
    linear_prediction
)

mse = mean_squared_error(
    y_test,
    linear_prediction
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    linear_prediction
)


print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==========================================================
# 9. ACTUAL VS PREDICTED GRAPH
# ==========================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    linear_prediction
)

plt.xlabel("Actual Absenteeism")
plt.ylabel("Predicted Absenteeism")

plt.title(
    "Linear Regression - Actual vs Predicted"
)

plt.tight_layout()
plt.show()


# ==========================================================
# 10. CREATE CLASSIFICATION TARGET
# ==========================================================

print("\n==============================")
print("CLASSIFICATION")
print("==============================")


# Median absenteeism

median_value = df[
    "Absenteeism time in hours"
].median()

print(
    "Median Absenteeism:",
    median_value
)


# Create classification target

# 0 = Low absenteeism
# 1 = High absenteeism

df["High_Absenteeism"] = (
    df["Absenteeism time in hours"]
    > median_value
).astype(int)


print("\nClass Distribution:")

print(
    df["High_Absenteeism"]
    .value_counts()
)


# ==========================================================
# 11. PREPARE CLASSIFICATION DATA
# ==========================================================

X = df.drop(
    [
        "Absenteeism time in hours",
        "High_Absenteeism"
    ],
    axis=1
)

y = df["High_Absenteeism"]


# Numerical features only

X = X.select_dtypes(
    include=["int64", "float64"]
)


# ==========================================================
# 12. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# 13. FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# 14. LOGISTIC REGRESSION
# ==========================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")


logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


logistic_prediction = logistic_model.predict(
    X_test_scaled
)


print(
    "Accuracy:",
    accuracy_score(
        y_test,
        logistic_prediction
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        logistic_prediction,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        logistic_prediction,
        zero_division=0
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        logistic_prediction,
        zero_division=0
    )
)


# ==========================================================
# 15. DECISION TREE
# ==========================================================

print("\n==============================")
print("DECISION TREE")
print("==============================")


decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)


tree_prediction = decision_tree.predict(
    X_test
)


print(
    "Accuracy:",
    accuracy_score(
        y_test,
        tree_prediction
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        tree_prediction,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        tree_prediction,
        zero_division=0
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        tree_prediction,
        zero_division=0
    )
)


# ==========================================================
# 16. RANDOM FOREST
# ==========================================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")


random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)


forest_prediction = random_forest.predict(
    X_test
)


print(
    "Accuracy:",
    accuracy_score(
        y_test,
        forest_prediction
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        forest_prediction,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        forest_prediction,
        zero_division=0
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        forest_prediction,
        zero_division=0
    )
)


# ==========================================================
# 17. CONFUSION MATRIX - RANDOM FOREST
# ==========================================================

cm = confusion_matrix(
    y_test,
    forest_prediction
)

print("\nRandom Forest Confusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()
plt.show()


# ==========================================================
# 18. FEATURE IMPORTANCE
# ==========================================================

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")


importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance":
        random_forest.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print(importance)


# ==========================================================
# 19. FEATURE IMPORTANCE GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Important Features"
)

plt.tight_layout()
plt.show()


# ==========================================================
# 20. MODEL COMPARISON
# ==========================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")


comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_score(
            y_test,
            logistic_prediction
        ),

        accuracy_score(
            y_test,
            tree_prediction
        ),

        accuracy_score(
            y_test,
            forest_prediction
        )
    ],

    "Precision": [
        precision_score(
            y_test,
            logistic_prediction,
            zero_division=0
        ),

        precision_score(
            y_test,
            tree_prediction,
            zero_division=0
        ),

        precision_score(
            y_test,
            forest_prediction,
            zero_division=0
        )
    ],

    "Recall": [
        recall_score(
            y_test,
            logistic_prediction,
            zero_division=0
        ),

        recall_score(
            y_test,
            tree_prediction,
            zero_division=0
        ),

        recall_score(
            y_test,
            forest_prediction,
            zero_division=0
        )
    ],

    "F1 Score": [
        f1_score(
            y_test,
            logistic_prediction,
            zero_division=0
        ),

        f1_score(
            y_test,
            tree_prediction,
            zero_division=0
        ),

        f1_score(
            y_test,
            forest_prediction,
            zero_division=0
        )
    ]

})


print(comparison)


# ==========================================================
# 21. BEST MODEL
# ==========================================================

best_model = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print("\n==============================")
print("BEST CLASSIFICATION MODEL")
print("==============================")

print(
    "Model:",
    best_model["Model"]
)

print(
    "Accuracy:",
    best_model["Accuracy"]
)

print(
    "Precision:",
    best_model["Precision"]
)

print(
    "Recall:",
    best_model["Recall"]
)

print(
    "F1 Score:",
    best_model["F1 Score"]
)


# ==========================================================
# 22. FINAL OUTPUT
# ==========================================================

print("\n==========================================")
print("MACHINE LEARNING ANALYSIS COMPLETED")
print("==========================================")

print("""
Techniques Used:

1. Data Cleaning
2. Correlation Analysis
3. Correlation Heatmap
4. Feature Engineering
5. Linear Regression
6. Logistic Regression
7. Decision Tree
8. Random Forest
9. Model Evaluation
10. Confusion Matrix
11. Feature Importance
12. Model Comparison
""")