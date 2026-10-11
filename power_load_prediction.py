
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("POWER LOAD TYPE PREDICTION")
print("=" * 60)

file_path = "data/load_data.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Dataset not found. Place load_data.csv inside the data folder."
    )

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nLoad type distribution:")
print(df["Load_Type"].value_counts())

df["Date_Time"] = pd.to_datetime(
    df["Date_Time"],
    dayfirst=True,
    errors="coerce"
)


df = df.dropna(subset=["Date_Time", "Load_Type"]).copy()

df["Load_Type"] = df["Load_Type"].astype(str).str.strip()

df = df.sort_values("Date_Time").reset_index(drop=True)

print("\nDataset after preprocessing:")
print("Shape:", df.shape)
print("Date range:", df["Date_Time"].min(),
      "to", df["Date_Time"].max())

print("Duplicate rows:", df.duplicated().sum())

df["Hour"] = df["Date_Time"].dt.hour
df["Day"] = df["Date_Time"].dt.day
df["Month"] = df["Date_Time"].dt.month
df["DayOfWeek"] = df["Date_Time"].dt.dayofweek
df["DayOfYear"] = df["Date_Time"].dt.dayofyear

print("\nTime-based features created.")

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Load_Type",
    hue="Load_Type",
    legend=False
)

plt.title("Distribution of Load Types")
plt.xlabel("Load Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/load_type_distribution.png")
plt.show()

if "Usage_kWh" in df.columns:
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="Usage_kWh",
        bins=40,
        kde=True
    )

    plt.title("Distribution of Electricity Usage")
    plt.xlabel("Usage in kWh")
    plt.tight_layout()
    plt.savefig("outputs/usage_distribution.png")
    plt.show()

if "Usage_kWh" in df.columns:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Load_Type",
        y="Usage_kWh",
        hue="Load_Type",
        legend=False
    )

    plt.title("Electricity Usage by Load Type")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("outputs/usage_by_load_type.png")
    plt.show()

numeric_df = df.select_dtypes(include=np.number)

if not numeric_df.empty:
    plt.figure(figsize=(11, 7))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.show()


target = "Load_Type"

X = df.drop(columns=[target, "Date_Time"])

y = df[target]

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

train_mask = (
    (df["Date_Time"].dt.year == 2018)
    & (df["Date_Time"].dt.month < 12)
)

test_mask = (
    (df["Date_Time"].dt.year == 2018)
    & (df["Date_Time"].dt.month == 12)
)

if not train_mask.any() or not test_mask.any():
    raise ValueError(
        "Expected training data from January-November 2018 "
        "and test data from December 2018. "
        "Check the dataset date range."
    )

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("\nTraining period:")
print(df.loc[train_mask, "Date_Time"].min(),
      "to", df.loc[train_mask, "Date_Time"].max())

print("\nTesting period:")
print(df.loc[test_mask, "Date_Time"].min(),
      "to", df.loc[test_mask, "Date_Time"].max())

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nTraining Random Forest...")

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

print("Random Forest training completed!")

print("\nTraining Logistic Regression...")

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

logistic_model.fit(X_train, y_train)

lr_predictions = logistic_model.predict(X_test)

print("Logistic Regression training completed!")

def evaluate_model(name, y_true, predictions):
    print("\n" + "=" * 60)
    print(name.upper(), "EVALUATION")
    print("=" * 60)

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true, predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true, predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true, predictions,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        predictions,
        zero_division=0
    ))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": f1
    }


rf_metrics = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

lr_metrics = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    rf_predictions,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues", xticks_rotation=30)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

comparison = pd.DataFrame([
    rf_metrics,
    lr_metrics
])

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# Compare macro F1-scores
plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison,
    x="Model",
    y="Macro_F1",
    hue="Model",
    legend=False
)

plt.title("Model Comparison - Macro F1-score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.show()

rf_classifier = random_forest_model.named_steps["classifier"]

fitted_preprocessor = (
    random_forest_model.named_steps["preprocessor"]
)

feature_names = fitted_preprocessor.get_feature_names_out()

importance = pd.Series(
    rf_classifier.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance.head(10))

top_features = importance.head(10).sort_values()

plt.figure(figsize=(9, 6))

top_features.plot(kind="barh")

plt.title("Top 10 Important Features - Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()

importance.to_csv("outputs/feature_importance.csv")

results = pd.DataFrame({
    "Date_Time": df.loc[test_mask, "Date_Time"].values,
    "Actual_Load_Type": y_test.values,
    "Random_Forest_Prediction": rf_predictions,
    "Logistic_Regression_Prediction": lr_predictions
})

results.to_csv(
    "outputs/load_type_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print("outputs/load_type_predictions.csv")

best_model_name = comparison.sort_values(
    "Macro_F1",
    ascending=False
).iloc[0]["Model"]

summary_path = "outputs/summary_report.txt"

with open(summary_path, "w", encoding="utf-8") as report:
    report.write("POWER LOAD TYPE PREDICTION REPORT\n")
    report.write("=" * 40 + "\n\n")

    report.write(f"Dataset shape: {df.shape}\n")
    report.write(f"Training records: {len(X_train)}\n")
    report.write(f"Testing records: {len(X_test)}\n")

    report.write("\nTraining period:\n")
    report.write(
        f"{df.loc[train_mask, 'Date_Time'].min()} to "
        f"{df.loc[train_mask, 'Date_Time'].max()}\n"
    )

    report.write("\nTesting period:\n")
    report.write(
        f"{df.loc[test_mask, 'Date_Time'].min()} to "
        f"{df.loc[test_mask, 'Date_Time'].max()}\n"
    )

    report.write("\nModel comparison:\n")
    report.write(comparison.to_string(index=False))

    report.write(f"\n\nBest model by macro F1-score: {best_model_name}\n")

print("\nSummary report saved to outputs/summary_report.txt")

print("\n" + "=" * 60)
print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("Check the outputs folder for results and graphs.")
print("=" * 60)
