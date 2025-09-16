# 🩺 Diabetes Analysis Using Python, Excel & Power BI  

This project focuses on analyzing **diabetes data** from NHANES using a mix of **Python**, **Excel**, and **Power BI**.  
The goal was simple: clean the raw data, explore it, and build a dashboard that shows real insights on diabetes trends.  

I didn’t use any medication-related columns in this project. The focus stayed on core diabetes indicators like glucose, HbA1c, BMI, and demographics. In the future, I might extend the work to include medication effects, but that was beyond the scope here.   

## Key Steps  

- **Data Cleaning, Transformation & Imputation**  
  - I started by cleaning and preparing the dataset using **Python (Pandas, NumPy, Scikit-learn)** and some **Excel** for quick checks. This involved fixing data types, removing duplicates, and making sure missing values didn’t mess up the analysis.  
  - For **numeric features**, I used **Iterative Imputer**, which does a better job than just filling with the mean because it estimates values based on other related features.  
  - **Categorical features** were handled by filling missing values with the **mode**, so nothing important was lost.  
  - Excel helped a lot for quick sanity checks and pivot tables to make sure everything looked reasonable.  

- **Exploratory Data Analysis (EDA)**  
  - I dug into patterns in blood sugar, BMI, and age to see what stood out.  
  - I also checked how diabetes prevalence changes by **gender** and different **age groups**, which helped spot trends and potential risk factors.  

- **Feature Engineering**  
  - Created new variables like **diabetes status** from glucose/HbA1c readings and grouped ages into categories to make comparisons easier.  
  - This was done alongside cleaning and imputation so that the dataset was ready for deeper analysis or any future modeling.  

- **Dashboard Development**  
  - Built interactive visuals in **Power BI** to make the data easy to explore.  
  - Used **DAX** to calculate things like prevalence rates, average BMI, and glucose trends over time.  
  - The final dashboard shows overall diabetes distribution, highlights key risk factors, and makes it easy to see patterns at a glance.



---

## Project Directory Structure


| Folder / File                                | Description                                              |
| -------------------------------------------- | -------------------------------------------------------- |
| **`data/`**                                  | Cleaned dataset used for analysis and dashboard          |
| **`dax/`**                                   | DAX measures and calculated columns for Power BI         |
| ├── `dax_formulas.txt`                       | List of all DAX formulas used                            |
| **`excel_project/`**                         | Excel-based analysis and pivot dashboards                |
| **`notebooks/`**                             | Jupyter notebooks for cleaning and EDA                   |
| ├── `diabetes_analysis.ipynb`                | Data cleaning, preprocessing, and exploratory analysis   |
| └── `diabetes_prediction.ipynb`              | Modeling and prediction of diabetes outcomes             |
| **`powerbi/`**                               | Power BI dashboards                                      |
| └── `diabetes_analysis.pbix`                 | Interactive Power BI report                              |
| **`reports/`**                               | Final summary exports                                    |
| └── `diabetes_analysis.pdf`                  | PDF export of main findings                              |
| **`report_screenshots/`**                    | Screenshots of Power BI visuals                          |
| ├── `clinical_indicators_report.PNG`                | Clinical indicators (glucose, BMI, HbA1c)                |
| ├── `demographics_&_risk_Factors_report.PNG`| Demographics and risk factor breakdown                   |
| └── `diabetes_overview_report.PNG`          | Overall summary of diabetes prevalence and trends        |
| **`requirements.txt`**                       | Python dependencies for notebooks                        |





---

## 📚 Table of Contents

- [📌 Project Overview](#project-overview)
- [🎯 Objectives](#objectives)
- [❓ Business Questions & Answers](#business-questions--answers)
- [🧩 Problem Statement](#problem-statement)
- [📂 Dataset](#dataset)
- [🧹 Data Processing, Cleaning & Imputation](#data-processing-cleaning--imputation)
  - [🔑 Key Steps](#key-steps)
- [🛠️ Tools & Technologies](#tools--technologies)
- [🧠 Skills Used](#skills-used)
- [📊 Excel Diabetes Analysis Overview](#excel-diabetes-analysis-overview)
  - [📸 Project Screenshots](#project-screenshots)
- [🐍 Python Overview](#python-overview)
  - [🐍 Python Data Cleaning and Preparation](#python-data-cleaning-and-preparation)
  - [🐍 Python Code Overview: notebooks](#python-code-overview-diabetes_analysisipynb--diabetes_mlipynb)
  - [🔍 Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [📈 Data Aggregation & Summary](#data-aggregation--summary)
  - [📊 Python Visualizations](#python-visualizations)
  - [📊 Predictive Modeling for Diabetes Risk](#python-predictive-modeling-for-diabetes-risk)
    - [🏆 Best Model: LightGBM](#best-model-lightgbm)
- [📊 Power BI Dashboard — Diabetes Analysis](#power-bi-dashboard--diabetes-analysis)
  - [📊 Report Structure](#report-structure)
  - [📊 Page Details & Visuals](#page-details--visuals)
  - [📊 Insights & Key Findings](#insights--key-findings--diabetes-analysis)
  - [📊 Business Recommendations](#business-recommendations)
  - [📊 Visual Types Summary](#visual-types-summary--diabetes-power-bi-dashboard)
  - [📊 Diabetes Power BI Report Previews](#diabetes-power-bi-report-previews)
  - [📊 Download the Full Power BI Report](#download-the-full-power-bi-report)
- [🧮 DAX Overview](#dax-overview)
  - [🧮 Key DAX](#key-dax)
  - [🧮 Key Calculated Columns](#key-calculated-columns)
- [📘 Visuals & Dashboard Summary](#visuals--dashboard-summary)
- [⚙️ Installation](#installation)
- [🙋‍♂️ Author](#author)

---

## Project Overview  
This project explores the **analysis of diabetes prevalence and risk factors** using Python,Excel and Power BI.  
It is based on real-world health survey data (NHANES) to identify patterns in blood sugar levels, demographics, and overall health indicators.  

---

## Objectives  

- **Data Preparation**  
  Use Python (Pandas, NumPy) and Excel to clean and prepare NHANES data.  

- **Health Insights**  
  Explore diabetes prevalence across age, gender, and demographics.  

- **Clinical Relationships**  
  Identify patterns between BMI, glucose, and HbA1c levels.  

- **Dashboard Delivery**  
  Build an interactive Power BI dashboard to communicate findings clearly.  

---

## Business Questions & Answers  

1. **What is the prevalence of diabetes across age groups and gender?**  
   Diabetes becomes more common with age, especially after 45. Men show slightly higher rates than women, though the gap evens out in older groups.  

2. **How do clinical indicators (BMI, glucose, HbA1c, insulin) relate to diabetes risk?**  
   High BMI, glucose, and HbA1c levels are closely tied to diabetes. Obesity stands out as a major risk factor, with insulin resistance adding further evidence.  

3. **How do demographic and health factors impact diabetes trends?**  
   Age, gender, and BMI categories strongly influence outcomes. Older adults and those who are overweight or obese face the highest risk.  

4. **What overall health patterns emerge among people with diabetes?**  
   People with diabetes tend to have higher blood pressure, cholesterol, and BMI, and are more likely to be on multiple medications. This points to diabetes often overlapping with other chronic conditions.  

---

## Problem Statement  

Diabetes is a growing public health concern worldwide, with rising cases linked to age, obesity, and lifestyle factors.  
Understanding its prevalence and associated health indicators is essential for prevention and management.  

This project uses NHANES data to examine how demographics and clinical measures relate to diabetes status.  
The aim is to generate insights that could support better awareness and decision-making.  

---

## Dataset  

Core NHANES files included in this project:  

- 📝 **Questionnaire**: Self-reported health and diabetes diagnosis  
- 💊 **Medications**: Prescription details and diabetes-related drugs (merged but not used in analysis)  
- 🧪 **Laboratory (Labs)**: Biomarkers such as glucose, insulin, and HbA1c  
- 🧍 **Examinations**: Physical measures including BMI and blood pressure  
- 🍽️ **Dietary Data**: Nutrition and food consumption patterns  
- 👥 **Demographics**: Age, gender, ethnicity, and socioeconomic details  

📌 **Note**: Dataset sourced from the official NHANES database on [Kaggle](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey/data)  


---

## Data Processing & Cleaning  

- ✅ **Handled Missing Values** – Dropped or imputed incomplete records to improve consistency  
- ✅ **Standardized Data Types** – Fixed numeric, categorical, and date fields for analysis  
- ✅ **Feature Engineering**:  
  - Merged NHANES source files into a single dataset (**Diabetes_Analysis.csv**)  
  - Created new variables (e.g., diabetes status from HbA1c/glucose, age groups)  
  - Included medication data during merging but excluded it from the analysis stage  
  - - Combined cleaned tables into a single analysis dataset ready for visualization

---

### Tools & Technologies  
- **Python (Pandas, NumPy, Matplotlib, Seaborn)** – Data cleaning, transformation, and exploratory analysis  
- **Excel** – Quick validation, pivot checks, and additional summaries  
- **Power BI** – Interactive dashboards using **DAX measures**  
- **Jupyter Notebook** – Step-by-step documentation and exploration  

---

## Skills Used  
- Data Cleaning & Preprocessing (Python, Pandas, Excel)  
- Feature Engineering for Health Data  
- Exploratory Data Analysis (EDA) & Visualization  
- Power BI Dashboard Development with DAX  
- Interpreting Demographic & Clinical Health Indicators  
- Git & GitHub Version Control  


---

## Excel Diabetes Analysis Overview

This section provides a quick look at the **Excel workflow** used in the **Diabetes Analysis Project**.

The project workflow is organized into:

- **Raw Data**
- **Cleaned Data**
- **Descriptive Analysis**
- **Pivot Tables**
- **Dashboard**

Each section demonstrates real-world data analysis steps, from cleaning to visualization, entirely in Excel.

### Project Screenshots

- **Raw Data**: ![data](report_screenshot/data.PNG)  
- **Cleaned Data**: ![data_cleaned](report_screenshot/data_cleaned.PNG)  
- **Descriptive Analysis 1**: ![descriptive_analysis](report_screenshot/descriptive_analysis.PNG)  
- **Descriptive Analysis 2**: ![descriptive_analysis2](report_screenshot/descriptive_analysis2.PNG)  
- **Descriptive Analysis 3**: ![descriptive_analysis3](report_screenshot/descriptive_analysis3.PNG)  
- **Pivot Table**: ![pivot_table](report_screenshot/pivot_table.png)  
- **Dashboard**: ![dashboard](report_screenshot/dashboard.PNG)

| Descriptive Analysis | Pivot Table | Dashboard |
|-------------------|-----------------------|----------------------|
| ![Descriptive Analysis](https://github.com/rotimi2020/nhanes-diabetes-analysis-prediction/blob/main/excel_project/report_screenshot/descriptive_analysis.PNG) | ![Pivot Table](https://github.com/rotimi2020/nhanes-diabetes-analysis-prediction/blob/main/excel_project/report_screenshot/pivot_table.png) | ![Dashboard](https://github.com/rotimi2020/nhanes-diabetes-analysis-prediction/blob/main/excel_project/report_screenshot/dashboard.PNG) |

> You can view all screenshots in the `report_screenshot` folder for a complete visual overview of the project.


---

## Python Overview 

This project leverages **Python** and **Jupyter Notebook** for in-depth analysis, feature engineering, and data visualization tasks related to **Diabetes Analysis**. Python scripts were used to:

- Explore and understand the structure and behavior of the NHANES diabetes dataset  
- Handle missing values by imputing or removing null entries  
- Convert object-type columns to appropriate numeric or categorical types for accurate analysis  
- Perform initial statistical validation and sanity checks on clinical measures such as glucose, BMI, blood pressure, and triglycerides  
- Visualize key patterns in diabetes prevalence, risk factors, and relationships between demographic and clinical variables  

Python enabled a smooth and efficient transformation pipeline that prepared the cleaned dataset (`nhanes_analysis.csv`) for visualization in Power BI. It supported logical consistency checks and helped validate the relationships between health indicators, demographics, and diabetes status before dashboard development.

---


### Python Data Cleaning and Preparation

The script `diabetes_analysis.ipynb` is a Python notebook developed to analyze and explore the **NHANES Diabetes** dataset.  
The data was sourced from multiple NHANES tables and merged into a unified dataset for analysis.

Key steps in the notebook include:

- **Data Exploration:**  
  Reviewing the structure of the data, checking value distributions, and identifying trends across demographic variables, clinical measurements, and diabetes status.

- **Data Preparation:**  
  Merging different tables (demographics, lab results, physical measures), formatting columns, handling missing values, and ensuring the dataset is analysis-ready.
  
- **KPI Calculation:**  
  Deriving key metrics like average glucose, BMI, diabetes prevalence, and other clinical indicators to support insights.

- **Visualization:**  
  Using charts to show glucose levels, BMI distribution, diabetes status patterns, and relationships between demographics and clinical measures.

This notebook acts as a solid base for generating insights and building visual dashboards in tools like Power BI.


---

## Python Code Overview: diabetes_analysis.ipynb & diabetes_ml.ipynb
> 📌 This section shows all core data loading, cleaning, merging, KPI preparation, and machine learning code used in the two notebooks:  
> - `diabetes_analysis.ipynb` — for exploratory data analysis and visualization  
> - `diabetes_ml.ipynb` — for predictive modeling and ML tasks  
>
> For the complete notebooks with full analysis, visualizations, and machine learning workflows, view the original files:  
> 👉 [diabetes_analysis.ipynb](path-to-notebook)  
> 👉 [diabetes_ml.ipynb](path-to-notebook)  
>
> ⚠️ Note: The code below represents only a part and section of the full notebooks.


```python
# Import required libraries
import numpy
import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
demographic = pd.read_csv(data_path + 'demographic.csv', encoding="latin1")
diet = pd.read_csv(data_path + 'diet.csv')
examination = pd.read_csv(data_path + 'examination.csv', encoding="latin1")
lab = pd.read_csv(data_path + 'labs.csv', encoding="latin1")
medication = pd.read_csv(data_path + 'medications.csv', encoding="latin1")
questionnaire = pd.read_csv(data_path + 'questionnaire.csv', encoding="latin1")

```

---

## 🔍 Exploratory Data Analysis (EDA)

```python
# Basic info
print(df.shape)          # dataset size
print(df.columns)        # column names
print(df.info())         # data types and non-null counts
print(df.describe())     # statistical summary

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age_Imputed'], bins=20, kde=True)
plt.title('Age Distribution (Imputed)', fontsize=14, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12, 14))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Top Features")
plt.show()
```

---

## 📈 Data Aggregation & Summary

```python
# Create obesity status categories
df['Obesity_Status'] = pd.cut(
    df['BMXBMI_imputed'],
    bins=[0, 24.9, 29.9, float('inf')],
    labels=['Non-Obese', 'Overweight', 'Obese']
)

# Risk Level Distribution
print(df['Risk_Level'].value_counts())

# Obesity Status Distribution
print(df['Obesity_Status'].value_counts())

```

---

## 📊 Visualizations

```python
# BMI Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['BMI_Imputed'], bins=20, kde=True)
plt.title('BMI Distribution (Imputed)', fontsize=14, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Diabetes by Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Gender', hue='Diabetes_Status')
plt.title('Diabetes Prevalence by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Diabetes Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Diabetes by Race
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='Race', hue='Diabetes_Status')
plt.title('Diabetes Prevalence by Race/Ethnicity', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Race/Ethnicity')
plt.legend(title='Diabetes Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
imputed_numeric_features = [col for col in df.columns if col.endswith("_Imputed") and col != 'Income_Code_Imputed']
correlation_matrix = df[imputed_numeric_features].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix (Imputed Values)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---  

### Python Predictive Modeling for Diabetes Risk

The notebook `diabetes_prediction.ipynb` is a full Python workflow for analyzing, cleaning, and modeling diabetes data from NHANES.  
It combines statistical exploration, feature engineering, and machine learning to identify at-risk individuals and support data-driven healthcare decisions.

**Highlights of the workflow:**

- **Exploratory Data Analysis (EDA):**  
  Diabetes prevalence (~9%) was analyzed along with key predictors like glucose, age, BMI, and waist circumference.  
  Feature relationships, including the high correlation between BMI and waist circumference (0.94), were explored. Visualizations guided feature selection and highlighted risk patterns.

- **Data Cleaning and Preparation:**  
  Missing clinical values were imputed, demographic codes standardized, and data consistency ensured across all survey variables.

- **Feature Engineering & Selection:**  
  Correlated variables were retained where clinically relevant. Random Forest identified top predictors: `Glucose_Imputed`, `Age`, `Waist_Circumference_Imputed`, `BMI_Imputed`, and `Triglycerides_Imputed`, providing interpretable, clinically meaningful insights.

- **Machine Learning Modeling:**  
  Random Forest, Gradient Boosting, XGBoost, and LightGBM were evaluated.  
  LightGBM performed best, balancing **screening recall (0.75)** and **diagnostic precision (0.851)**.  
  Thresholds were optimized for dual purposes:
  - **Screening Mode (0.70):** Detects most at-risk individuals for early intervention.  
  - **Diagnostic Mode (0.90):** Offers high-confidence predictions for treatment decisions.

- **Model Interpretation & Clinical Insights:**  
  Feature importance and SHAP analyses confirm that model decisions are driven by key clinical variables, not demographic proxies.  
  Recommendations include targeted monitoring of glucose, age-related risk, BMI, and waist circumference, with gender-inclusive strategies.

- **Future Enhancements:**  
  Plans include SMOTE for better minority detection, SHAP dashboards for explainability, and continuous monitoring pipelines. The framework is also adaptable for other preventive health applications.

This notebook demonstrates a practical, human-centered approach to diabetes prediction, blending statistical rigor with clinical utility for actionable insights.

**Note on Feature Selection:**  
The correlation plot shows that BMI and waist circumference are highly correlated (0.94). Despite this, both features were **intentionally retained** because tree-based models like LightGBM and Random Forest handle multicollinearity well, so model performance is not affected.  

Feature selection was guided by **domain knowledge** rather than automated methods like RFE. This ensures that clinically meaningful predictors—those known to influence diabetes risk—remain in the model, providing **interpretability** and **trustworthiness** for real-world healthcare use.


---

#### Model Performance Summary

| Algorithm              | Threshold Type                     | Recall | Precision | F1 Score | ROC AUC | Notes                                                                 |
|------------------------|-----------------------------------|--------|-----------|----------|---------|-----------------------------------------------------------------------|
| Gradient Boosting      | Default (0.50)                     | 0.34   | 0.731     | 0.464    | 0.665   | Good baseline, moderate precision, low recall for diabetes detection |
| Gradient Boosting      | Best Threshold (0.282)             | 0.604  | 0.540     | 0.570    | 0.78    | Balanced performance after optimization                               |
| Gradient Boosting      | Screening (0.08)                   | 0.847  | 0.334     | 0.479    | 0.852   | High sensitivity for early detection, more false positives           |
| Gradient Boosting      | Diagnostic (0.55)                  | 0.292  | 0.808     | 0.429    | 0.643   | High precision for confirmation, lower sensitivity                    |
| XGBoost                | Default (0.50)                     | 0.312  | 0.726     | 0.437    | 0.651   | Similar baseline performance to Gradient Boosting                     |
| XGBoost                | Best Threshold (0.36)              | 0.507  | 0.640     | 0.566    | 0.741   | Improved balance with optimized threshold                              |
| XGBoost                | Screening (0.12)                   | 0.806  | 0.352     | 0.489    | 0.84    | High recall for at-risk patients                                       |
| XGBoost                | Diagnostic (0.55)                  | 0.236  | 0.773     | 0.362    | 0.615   | High precision, lower recall                                           |
| Random Forest          | Default (0.50)                     | 0.208  | 0.789     | 0.330    | 0.602   | Strong precision, low sensitivity                                      |
| Random Forest          | Best Threshold (0.281)             | 0.569  | 0.566     | 0.567    | 0.766   | Balanced after threshold optimization                                  |
| Random Forest          | Screening (0.10)                   | 0.854  | 0.311     | 0.456    | 0.847   | High sensitivity for screening                                        |
| Random Forest          | Diagnostic (0.48)                  | 0.215  | 0.756     | 0.335    | 0.605   | High confidence predictions, lower recall                               |
| LightGBM               | Default (0.50)                     | 0.840  | 0.306     | 0.449    | 0.84    | Strong recall for early detection                                      |
| LightGBM               | Best Threshold (0.805)             | 0.646  | 0.541     | 0.589    | 0.80    | Balanced performance                                                    |
| LightGBM               | Screening (0.70)                   | 0.750  | 0.406     | 0.527    | 0.829   | Optimal for population-level screening                                  |
| LightGBM               | Diagnostic (0.90)                  | 0.278  | 0.851     | 0.419    | 0.637   | Optimal for high-confidence treatment decisions                         |

> **Note:**  
- Tree-based models handle multicollinearity effectively (e.g., BMI vs waist circumference).  
- Screening thresholds prioritize recall to catch most at-risk patients.  
- Diagnostic thresholds prioritize precision to reduce false positives.  
- LightGBM is the best overall model for dual-purpose diabetes prediction, balancing screening and diagnostic performance.  
- Average Precision (AUC-PR) is consistent at 0.618 across models, indicating robust detection of minority diabetic cases.  
- Manual threshold tuning allows each model to adapt to population-level screening and individual diagnostic needs.

---

## 🏆 Best Model: LightGBM

**Why LightGBM Stands Out:**  
LightGBM was selected for **superior clinical utility**, balancing both screening and diagnostic needs:

- **Screening Recall:** 0.750 – detects 75% of at-risk individuals, a **184% improvement** over baseline thresholds.  
- **Diagnostic Precision:** 0.851 – provides 85% confidence in positive predictions, minimizing unnecessary treatments.  
- **F1 Scores:** Maintains strong balance for both screening (0.527) and diagnostic (0.419) applications.  
- **Average Precision (AUC-PR):** 0.618 – highest among all models evaluated.  

### 📊 Model Comparison Snapshot

| Model              | Screening Recall | Diagnostic Precision | Verdict                        |  
|-------------------|----------------|--------------------|--------------------------------|  
| LightGBM           | 0.750          | 0.851              | ✅ Best Overall                 |  
| Gradient Boosting  | 0.847          | 0.808              | Good screening, weaker diagnostics |  
| XGBoost            | 0.806          | 0.773              | Moderate both applications     |  
| Random Forest      | 0.854          | 0.756              | Strong screening only          |  

**Key Insight:** LightGBM’s dual-threshold optimization allows a single model to function as both a sensitive **screening tool** and a precise **diagnostic aid**, making it ideal for practical diabetes risk prediction.

---


## Power BI Dashboard — Diabetes Analysis

This Power BI report delivers a complete analysis of **diabetes-related data**, covering clinical measures, participant demographics, risk factors, and prevalence. It includes 5 dashboard pages that highlight trends in glucose levels, BMI, diabetes status, medication use, and demographic patterns—supporting smarter decisions around healthcare interventions and population health management.

---

<h2 id="report_structure"> Report Structure </h2>

This Power BI report explores **diabetes data** using five interactive dashboards. Each page uncovers meaningful insights into clinical indicators, risk factors, and participant profiles—turning raw data into actionable visual stories.

| Page # | Dashboard Title           | Description                                                                 |
|--------|---------------------------|-----------------------------------------------------------------------------|
| 1️⃣     | 📊 Diabetes Overview       | Overview of diabetes prevalence participant distribution |
| 2️⃣     | 👥 Demographics & Risk     | Age, gender, ethnicity, BMI, and risk factor analysis                       |
| 3️⃣     | 🩺 Clinical Measures       | Fasting glucose, HbA1c, blood pressure, and BMI trends                       |



---

## Page Details & Visuals — Diabetes Power BI Dashboard

#### 1️⃣ Diabetes Overview (Prevalence)

This section gives a snapshot of how diabetes affects different groups in the population:

- 📊 **100% Stacked Bar Column Chart:** Diabetes Prevalence by Gender – see how diabetes is distributed between men and women.  
- 📊 **Stacked Column Chart:** Diabetes Prevalence by Race/Ethnicity – shows differences in diabetes rates across racial groups.  
- 📊 **Clustered Column Chart:** Diabetes Prevalence by Age Group – breaks down prevalence by age bins (<20, 20–39, 40–59, 60+).

🌟 **Why it matters:**  
- Get a clear picture of overall diabetes prevalence.  
- Spot differences between genders, races, and age groups.  
- Identify which groups are most at risk.

---

#### 2️⃣ Demographics & Risk Factors

Here we look at socio-economic and family-related factors that may influence diabetes:

- 📊 **Clustered Column Chart:** Diabetes by Education Level – see how education correlates with diabetes prevalence.  
- 📊 **Stacked Bar Chart:** Diabetes by Income Group – compares prevalence across different income brackets.  
- 🍩 **Donut Chart:** Family History vs Diabetes Status – shows how having diabetic relatives impacts your own diabetes risk.  
- 🗺️ **Treemap/Pie Chart:** Risk Level Distribution – visualizes the share of low, medium, and high-risk individuals.

🌟 **Why it matters:**  
- Understand how education and income relate to diabetes.  
- See the influence of family history.  
- Keep track of overall risk distribution in the population.

---

#### 3️⃣ Clinical Indicators

This part dives into health measurements and how they relate to diabetes:

- 📊 **Clustered Bar Chart:** BMI by Diabetes Status – compares average BMI for diabetic vs non-diabetic participants.  
- 📊 **Clustered Column & Line Combo Chart:** Blood Pressure by Diabetes Status – columns show systolic BP, line shows diastolic BP.  
- 🔬 **Scatter Chart:** BMI & Glucose by Diabetes Status – shows the relationship between BMI and glucose levels.  
- 📊 **100% Stacked Column Chart:** Diabetes by Obesity Status – compares the share of diabetes across obesity categories.

🌟 **Why it matters:**  
- Keep an eye on important clinical indicators.  
- Spot patterns like higher BMI or glucose in diabetic participants.  
- Understand how obesity contributes to diabetes prevalence.

---

# 🔑 Insights & Key Findings — Diabetes Analysis

### 1. Demographics & Prevalence
- **Age is the strongest risk factor**: Diabetes prevalence jumps from **0.4% (0–20 yrs)** to **24.9% (60+ yrs)**, showing a clear age-driven risk increase.  
- **Gender gap is minimal**: Both men and women show similar prevalence (~8%), indicating that gender alone is not a major differentiator.  
- **Race disparities exist**: Non-Hispanic Black and Mexican-American groups show relatively higher diabetes cases compared to Asians and Whites.  
- **Education & income matter**: Lower education levels (less than high school) and low-income groups show disproportionately higher diabetes cases compared to college graduates and higher earners.  

---

### 2. Clinical & Health Indicators
- **Blood pressure**: Diabetic participants have significantly higher averages (**Systolic 130 vs 113; Diastolic 68 vs 63**).  
- **Lipids**: Triglycerides are much higher in diabetics (**163 vs 106**), while HDL (good cholesterol) is lower (**47 vs 54**). This aligns with known cardiovascular risks linked to diabetes.  
- **Obesity is a strong driver**:  
  - **Obese group:** 17.6% diabetic.  
  - **Overweight group:** 10.3% diabetic.  
  - **Non-obese group:** only 2.1% diabetic.  
  This shows obesity significantly elevates diabetes risk.  

---

### 3. Family & Risk Distribution
- **Family history is a major factor**: Participants with a family history of diabetes are more than **2x as likely** to have diabetes compared to those without.  
- **Risk level distribution**: About **37% of participants fall in the “High Risk” category**, meaning proactive intervention is critical to prevent future cases.  

---

# 💡 Business Recommendations

### Public Health & Awareness
- **Target older adults (40+)** with preventive screenings and education since prevalence rises sharply after 40.  
- **Promote lifestyle education** programs around obesity reduction, healthy diets, and physical activity—especially in communities with high obesity prevalence.  
- **Design culturally tailored interventions** for high-risk groups (e.g., Black and Mexican-American populations) to address disparities.  

### Healthcare & Clinical Action
- **Integrate family history into screening**: Anyone with a diabetic family background should be prioritized for regular glucose and BMI checks.  
- **Monitor cardiovascular markers**: High triglycerides, low HDL, and hypertension in diabetic patients signal the need for combined heart-health and diabetes care programs.  
- **Focus on education-linked risks**: Provide community clinics and health campaigns in low-education and low-income areas to close access gaps.  

### Policy & Preventive Strategy
- **Develop obesity-focused initiatives**: Tax incentives, subsidized fitness programs, and nutrition policies could lower obesity-driven diabetes rates.  
- **Early detection campaigns**: Use the “High Risk” population (37%) as the focus for cost-effective screening and prevention to reduce long-term healthcare costs.  

---

✅ **Overall Impact:**  
This analysis shows diabetes risk is **driven most by age, obesity, blood pressure, and family history**—with education, income, and race as contributing factors. Targeting these high-impact areas with **preventive programs, community-level education, and early screenings** can reduce future prevalence and improve population health outcomes.


---

### Visual Types Summary — Diabetes Power BI Dashboard

In this dashboard, I used different visual types to make the insights clear and easy to interpret:

| Visual Type         | How I Used It                                                                 |
|---------------------|--------------------------------------------------------------------------------|
| 100% Stacked Bar    | Showed how diabetes prevalence compares by **gender** and **obesity status**. |
| Stacked Column      | Highlighted diabetes counts across **race/ethnicity** and **income groups**.  |
| Clustered Column    | Compared side-by-side results for **age groups** and **education levels**.    |
| Clustered Bar       | Displayed averages like **BMI across diabetes vs non-diabetes groups**.       |
| Combo Chart         | Plotted **systolic and diastolic blood pressure** together for comparison.    |
| Scatter Plot        | Explored the relationship between **BMI and glucose** by diabetes status.     |
| Pie/Donut Chart     | Illustrated the effect of **family history** on diabetes outcomes.            |
| Treemap             | Broke down the population into **low, medium, and high risk levels**.         |
| Slicers             | Added filters for **diabetes status** and **gender** so users can interact.   |

Each chart was chosen to connect the data back to real health insights — from showing who’s most at risk, to visualizing how clinical factors like BMI and blood pressure tie into diabetes.


---

## 🩺 Diabetes Power BI Report Previews  

Below are sample preview images from the **Diabetes Analysis Power BI reports**.  
These visuals highlight key health insights, patterns in diabetes prevalence, demographics, and clinical indicators.  
Each report page is designed to make the data story clear and easy to explore.  

| Diabetes Overview | Demographic Insights | Clinical Indicators |
|-------------------|-----------------------|----------------------|
| ![Diabetes Overview](https://github.com/rotimi2020/nhanes-diabetes-analysis-prediction/blob/main/report_screenshots/diabetes_overview_report.PNG) | ![Demographic Insights](https://github.com/rotimi2020/nhanes-diabetes-analysis-prediction/blob/main/report_screenshots/demographics_&_risk_Factors_report.PNG) | ![Clinical Indicators](https://github.com/rotimi2020/nhanes-diabetes-analysis-prediction/blob/main/report_screenshots/clinical_indicators_report.PNG) |


---

### 📄 Download the Full Power BI Report  

You can **download the complete Diabetes Power BI report** in PDF format:  
**[Download PDF Report](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/diabetes_analysis/reports/diabetes_report.pdf)**


---

# DAX Overview

This project uses **Power BI and DAX** to analyze the **NHANES Diabetes dataset**.  
We created custom measures and calculated columns to track diabetes prevalence, BMI, glucose levels, demographics, and clinical insights.

---

## Key DAX

```sql
# Total Participants
Total Participants = COUNTROWS(diabetes_analysis)  
```

```sql
# Participants With Diabetes
With Diabetes = 
CALCULATE(
    COUNTROWS(diabetes_analysis),
    diabetes_analysis[Diabetes_Status] = "Diabetes"
)
```

```sql
# Participants Without Diabetes
Without Diabetes = 
CALCULATE(
    COUNTROWS(diabetes_analysis),
    diabetes_analysis[Diabetes_Status] = "Non_Diabetes"
)
```

```sql
# % Patients With Diabetes
% Patients with Diabetes = 
DIVIDE([With Diabetes], [Total Participants], 0)
```

```sql
# Average BMI
Avg BMI = AVERAGE(diabetes_analysis[BMI_Imputed])  
```

```sql
# Average Glucose
Avg Glucose = AVERAGE(diabetes_analysis[Glucose_Imputed])  
```

```sql
# Average HDL
Avg HDL = AVERAGE(diabetes_analysis[HDL_Imputed])
```

```sql
# Average Triglycerides
Avg Triglycerides = AVERAGE(diabetes_analysis[Triglycerides_Imputed])
```

---

## Key Calculated Columns

```sql
# BMI Category
BMI Category = 
SWITCH(
    TRUE(),
    diabetes_analysis[BMI_Imputed] < 18.5, "Under weight",
    diabetes_analysis[BMI_Imputed] < 25, "Normal weight",
    diabetes_analysis[BMI_Imputed] < 30, "Over weight",
    "Obese"
)
```

```sql
# Age Bin
Age Bin = 
SWITCH(
    TRUE(),
    diabetes_analysis[Age] >= 0 && diabetes_analysis[Age] <= 20, "0-20",
    diabetes_analysis[Age] >= 21 && diabetes_analysis[Age] <= 40, "21-40",
    diabetes_analysis[Age] >= 41 && diabetes_analysis[Age] <= 60, "41-60",
    diabetes_analysis[Age] > 60, "60+",
    BLANK()
)
```

```sql
# Race Short
Race Short = 
SWITCH(
    diabetes_analysis[Race],
    "Non-Hispanic White", "NH White",
    "Non-Hispanic Black", "NH Black",
    "Mexican American", "Mex-Am",
    "Other Hispanic", "Oth Hisp",
    "Other Race - Including Multi-Racial", "Other/Multi",
    diabetes_analysis[Race]
)
```

```sql
# Diabetes Status Label
Diabetes Status Label = 
IF(
    diabetes_analysis[Diabetes_Status] = "Diabetes",
    "Diabetes",
    "No Diabetes"
)
```

---



## Visuals & Dashboard Summary  

- Line and bar charts highlight trends in diabetes prevalence, BMI, and glucose levels across different groups.  
- KPI cards give a quick snapshot of total participants, percentage with diabetes, and key averages like BMI and glucose.  
- Slicers make the report interactive, letting you filter by demographics such as age, gender, and race to explore patterns more closely.  

📂 **Download Full DAX Code File**:  
[View on GitHub](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/diabetes_analysis/dax/dax_formulas.txt)  


---


<h2 id="installation"> ⚙️ Installation </h2>

To set up the project environment on your local machine, follow these steps:

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/rotimi2020/Data-Analyst-Portfolio.git
cd Data-Analyst-Portfolio/Diabetes_Analysis

```

### ✅ Step 2: Install Dependencies
Make sure Python 3.8 or later is installed. Then run:

```bash
pip install -r requirements.txt
```

---

<h2 id="author"> 🙋‍♂️ Author </h2>

**Rotimi Sheriff Omosewo**  
📧 Email: [omoseworotimi@gmail.com](mailto:omoseworotimi@gmail.com)  
📞 Contact: +234 903 441 1444  
🔗 LinkedIn: [linkedin.com/in/rotimi-sheriff-omosewo-939a806b](https://www.linkedin.com/in/rotimi-sheriff-omosewo-939a806b)  
📁 Project GitHub: [github.com/rotimi2020/Data-Analyst-Portfolio](https://github.com/rotimi2020/Data-Analyst-Portfolio)  

> 📌 **Note:** This **Diabetes Analysis** project is part of my Data Analytics Portfolio. It demonstrates skills in **data wrangling, statistical analysis, and business intelligence** using the NHANES dataset. Tools used include **Python (for analysis)** and **Power BI (for dashboarding and DAX modeling)**.


---
