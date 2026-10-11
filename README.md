# Power Load Type Prediction Using Machine Learning

## 1. Project Overview

The Power Load Type Prediction project uses machine learning techniques to classify electricity consumption patterns into different load categories based on historical power consumption data.

The project analyzes electricity usage, reactive power, power factor, CO₂ emissions, and time-related features to predict the load type.

The system classifies electricity consumption into three categories:

- **Light Load**
- **Medium Load**
- **Maximum Load**

Two machine learning algorithms, Random Forest and Logistic Regression, are trained and evaluated to compare their classification performance.

## 2. Objectives

- Analyze historical electricity consumption data.
- Perform data preprocessing and handle missing values.
- Conduct Exploratory Data Analysis (EDA) using visualizations.
- Extract useful time-based features from timestamps.
- Build machine learning classification models.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Compare different classification algorithms.
- Identify important features influencing load classification.

## 3. Technologies Used

- **Programming Language:** Python
- **IDE:** Visual Studio Code
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Dataset Format:** CSV

## 4. Dataset Description

The project uses `load_data.csv`, which contains 35,041 records and 9 original columns covering electricity consumption data from January to December 2018.

| Column | Description |
|---|---|
| `Date_Time` | Date and time of the observation |
| `Usage_kWh` | Electricity consumption in kWh |
| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive power consumption |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive power consumption |
| `CO2(tCO2)` | CO₂ emissions measurement |
| `Lagging_Current_Power_Factor` | Lagging current power factor |
| `Leading_Current_Power_Factor` | Leading current power factor |
| `NSM` | Number of seconds from midnight |
| `Load_Type` | Target classification label |

### Target Classes

- `Light_Load`
- `Medium_Load`
- `Maximum_Load`

## 5. Project Workflow

1. **Data Loading:** Load the CSV dataset using Pandas.
2. **Data Inspection:** Examine dataset dimensions, data types, missing values, and class distribution.
3. **Data Preprocessing:** Convert timestamps, remove invalid dates or missing target labels, and sort records chronologically.
4. **Feature Engineering:** Extract hour, day, month, day of the week, and day of the year.
5. **Exploratory Data Analysis:** Visualize load type distribution, electricity usage, and feature correlations.
6. **Data Splitting:** Use January to November 2018 for training and December 2018 for testing.
7. **Model Training:** Train Random Forest and Logistic Regression classifiers using preprocessing pipelines.
8. **Model Evaluation:** Calculate accuracy, macro precision, macro recall, macro F1-score, and classification reports.
9. **Model Comparison:** Compare the performance of both algorithms.
10. **Feature Importance:** Identify the most influential features in the Random Forest model.
11. **Result Export:** Save predictions, comparison metrics, and visualizations.

## 6. Machine Learning Algorithms

### Random Forest Classifier

Random Forest combines multiple decision trees to perform classification. It can learn nonlinear relationships between electricity consumption features and load categories.

### Logistic Regression

Logistic Regression is a classification algorithm used as a baseline for comparison. It estimates the probability that an observation belongs to each load category.

Both models are evaluated using the same chronological training and testing periods.

## 7. Evaluation Metrics

The following metrics are used to evaluate the models:

- **Accuracy:** Proportion of correctly classified observations.
- **Precision:** Proportion of correct predictions among all predictions for a class.
- **Recall:** Proportion of actual observations of a class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct classifications and misclassifications for each load category.

Macro-averaged precision, recall, and F1-score are used to give equal importance to each class.

## 8. Project Structure

```text
Power_Load_Prediction/
│
├── .venv/
│
├── data/
│   └── load_data.csv
│
├── outputs/
│   ├── load_type_distribution.png
│   ├── usage_distribution.png
│   ├── usage_by_load_type.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   ├── load_type_predictions.csv
│   └── summary_report.txt
│
├── power_load_prediction.py
├── requirements.txt
└── README.md
```

*Note: Files inside `outputs/` are generated when the Python program runs successfully.*

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- Visual Studio Code
- Python extension for VS Code

### Step 1: Clone the repository

If the project is hosted on GitHub:

```bash
git clone <your-repository-url>
cd Power_Load_Prediction
```

Alternatively, open the project folder directly in VS Code.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

For Windows PowerShell:

```powershell
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify the dataset location

Ensure the dataset is stored at:

```text
data/load_data.csv
```

### Step 6: Run the program

```bash
python power_load_prediction.py
```

The program prints the evaluation results in the terminal and generates the output files in the `outputs/` directory.

## 10. Expected Outputs

After successful execution, the project generates:

- Load type distribution graph
- Electricity usage distribution graph
- Electricity usage by load type graph
- Numerical feature correlation heatmap
- Random Forest confusion matrix
- Model comparison graph
- Feature importance graph
- Test-set predictions in CSV format
- Model comparison metrics in CSV format
- Summary report in text format

Actual performance metrics depend on the dataset, preprocessing, and model configuration.

## 11. Results and Conclusion

The project demonstrates how machine learning can be applied to historical electricity consumption data to classify power load types.

Random Forest and Logistic Regression are evaluated using a chronological holdout approach, with December 2018 reserved for testing. Their performance is compared using accuracy, macro precision, macro recall, and macro F1-score.

The final comparison helps determine which model performs better on the unseen test period. Feature importance analysis provides additional insight into the characteristics associated with load classification.

**Final model performance:** Add the actual evaluation metrics obtained after running the program.

## 12. Future Enhancements

- Experiment with additional classification algorithms.
- Perform hyperparameter tuning to improve performance.
- Develop an interactive dashboard for load prediction.
- Integrate real-time electricity consumption data.
- Deploy the trained model as a web application.
- Investigate time-series forecasting for future electricity demand.

## 13. Author

**Project:** Power Load Type Prediction Using Machine Learning

**Language:** Python

**Environment:** Visual Studio Code

