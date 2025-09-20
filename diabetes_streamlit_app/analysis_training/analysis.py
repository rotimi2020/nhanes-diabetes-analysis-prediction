#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
#
# -----------------------------------------------------------------
#                                             NHANES Diabetes Analysis - Data Processing Pipeline
# -----------------------------------------------------------------

# %%

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import set_option

# Import scikit-learn modules
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Configure display options
set_option('display.max_rows', 500)
set_option('display.max_columns', 500)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# %% [markdown]
# # NHANES Diabetes Data Analysis
#
# ## Project Overview
#
# This notebook documents the comprehensive data analysis for a project investigating factors related to diabetes. Using data from the National Health and Nutrition Examination Survey (NHANES), we will:
#
# 1.  **Import and Merge** the relevant data files from different NHANES components.
# 2.  **Clean and Preprocess** the data, handling missing values and recoding variables according to the codebook specifications
# 3.  **Engineer New Features** to create more powerful predictors that capture clinical insights and interactions.
# 4.  **Perform Exploratory Data Analysis (EDA)** to understand distributions, correlations, and potential relationships.
# 5.  **Create Visualizations** to communicate key findings and patterns in the data.
#
# This foundational work in data preparation, exploration, and feature engineering is essential for building effective predictive models in the next phase of the project.
#
# **Data Source:** [NHANES Dataset on Kaggle](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey/data)
#
# Let's begin by importing the necessary libraries and then loading our data.

# %%


# Load NHANES dataset components
data_path = 'C:/Users/HP/Desktop/Porfolio/Portfolio/nhanes/nhanes-diabetes-project/diabetes_analysis/diabetes_streamlit_app/analysis/'

demographic = pd.read_csv(data_path + 'demographic.csv', encoding="latin1")
diet = pd.read_csv(data_path + 'diet.csv')
examination = pd.read_csv(data_path + 'examination.csv', encoding="latin1")
lab = pd.read_csv(data_path + 'labs.csv', encoding="latin1")
medication = pd.read_csv(data_path + 'medications.csv', encoding="latin1")
questionnaire = pd.read_csv(data_path + 'questionnaire.csv', encoding="latin1")


# %%


# Preview dataset structures
print("=== Dataset Previews ===")

print("\nDemographic Data (First 5 rows):")
demographic.head()


# %%

# Display the first 5 rows of the Diet dataset
print("\nDiet Data (First 5 rows):")
diet.head()


# %%

# Display the first 5 rows of the Examination dataset
print("\nExamination Data (First 5 rows):")
examination.head()


# %%

# Display the first 5 rows of the Lab dataset
print("\nLab Data (First 5 rows):")
lab.head()


# %%

# Display the first 5 rows of the Medication dataset
print("\nMedication Data (First 5 rows):")
medication.head()


# %%

# Display the first 5 rows of the Questionnaire dataset
print("\nQuestionnaire Data (First 5 rows):")
questionnaire.head()


# %%


# Dataset summaries
print("\n" + "="*50)
print("DATASET SUMMARIES")
print("="*50)

datasets = {
    'questionnaire': questionnaire,
    'medication': medication,
    'lab': lab,
    'examination': examination,
    'diet': diet,
    'demographic': demographic
}

for name, dataset in datasets.items():
    print(f"\n{name.upper()} Overview:")
    print(f"Shape: {dataset.shape}")
    print("Unique values per column:")
    print(dataset.nunique().head(10))


# %%


# Select relevant features from each dataset
print("\nSelecting relevant features for analysis...")

demographic_cols = demographic[[
    "SEQN",        # Respondent sequence number (unique participant ID)
    "RIDAGEYR",    # Age in years
    "RIAGENDR",    # Gender
    "RIDRETH3",    # Race/Ethnicity
    "DMDEDUC2",    # Education Level
    "INDHHIN2"     # Household Income
]]

examination_cols = examination[[
    "SEQN",        # Respondent sequence number
    "BMXBMI",      # Body Mass Index
    "BMXWAIST",    # Waist Circumference
    "BPXSY1",      # Systolic BP
    "BPXDI1"       # Diastolic BP
]]

lab_cols = lab[[
    "SEQN",        # Respondent sequence number
    "LBXGLT",      # 2-Hr Glucose
    "LBDHDD",      # HDL Cholesterol
    "LBXTR"        # Triglycerides
]]

questionnaire_cols = questionnaire[[
    "SEQN",        # Respondent sequence number
    "DIQ010",      # Doctor's diagnosis of diabetes
    "MCQ300C"      # Family History
]]


# %%


# Merge datasets into unified dataframe
print("Merging datasets...")

df = demographic_cols.merge(questionnaire_cols, on="SEQN", how="inner", validate='one_to_one')
df = df.merge(examination_cols, on="SEQN", how="inner", validate='one_to_one')
df = df.merge(lab_cols, on="SEQN", how="inner", validate='one_to_one')

print(f"Merged dataset shape: {df.shape}")
df.head()


# %%
# Save the dataset for excel analysis
output_path = 'nhanes_merge.csv'
df.to_csv(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")
print(f"Final dataset shape: {df.shape}")

# %%


# Initial missing values analysis
print("Initial missing values analysis:")
missing = df.isnull().mean().sort_values(ascending=False)
print(missing.head(10))


# %%


# Handle special missing value codes
print("Processing special missing value codes...")

# Education: 7 = Refused, 9 = Don't Know → Convert to NaN
df['DMDEDUC2'].replace({7.0: np.nan, 9.0: np.nan}, inplace=True)

# Income: 77 = Refused, 99 = Don't Know → Convert to NaN
df['INDHHIN2'].replace({77.0: np.nan, 99.0: np.nan}, inplace=True)

# Family diabetes history: 7 = Refused, 9 = Don't Know → Convert to NaN
df['MCQ300C'].replace({7.0: np.nan, 9.0: np.nan}, inplace=True)

# Diabetes indicator: 3 = Borderline, 7/9 = Refused/Don't Know → Convert to NaN
df['DIQ010'].replace({3.0: np.nan, 7.0: np.nan, 9.0: np.nan}, inplace=True)


# %%


# Updated missing values analysis
print("Missing values after processing special codes:")
missing = df.isnull().mean().sort_values(ascending=False)
print(missing.head(10))


# %% [markdown]
# --------------------------
#                                             Feature Engineering and Data Imputation
# --------------------------

# %%


# Separate feature types for appropriate imputation
numeric_features = ['RIDAGEYR', 'BMXBMI', 'BMXWAIST', 'BPXSY1', 'BPXDI1', 
                    'LBXGLT', 'LBDHDD', 'LBXTR']
categorical_features = ['RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDHHIN2', 'MCQ300C']

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")


# %%


# Numeric feature imputation using Iterative Imputer
print("Imputing numeric features...")

imputer = IterativeImputer(max_iter=10, random_state=42)
num_df = df[numeric_features]
imputed_num = imputer.fit_transform(num_df)

for i, feature in enumerate(numeric_features):
    df[f'{feature}_missing'] = df[feature].isnull().astype(int)
    df[f'{feature}_imputed'] = imputed_num[:, i]


# %%


# Categorical feature imputation using mode
print("Imputing categorical features...")

for feature in categorical_features:
    df[f'{feature}_missing'] = df[feature].isnull().astype(int)
    if df[feature].notnull().any():
        most_common = df[feature].mode()[0]
        df[f'{feature}_imputed'] = df[feature].fillna(most_common)
    else:
        df[f'{feature}_imputed'] = df[feature].fillna(1)


# %%


# Create descriptive labels for analysis
print("Creating descriptive labels...")

# Mapping dictionaries
gender_map = {1: 'Male', 2: 'Female'}
race_map = {
    1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',
    4: 'Non-Hispanic Black', 6: 'Non-Hispanic Asian', 7: 'Other Race'
}
education_map = {
    1: 'Less than 9th grade', 2: '9-11th grade', 3: 'High school graduate',
    4: 'Some college or AA degree', 5: 'College graduate'
}

# income_imputed = {
#     1: '0-4999', 2: '5000-9999', 3: '10000-14999', 4: '15000-19999',
#     5: '20000-24999', 6: '25000-34999', 7: '35000-44999', 8: '45000-54999',
#     9: '55000-64999', 10: '65000-74999', 11: '75000-99999', 12: '100000+',
#     13: 'Over 200000'
# }

income_map = { 1.0: '2500', 2.0: '7500', 3.0: '12500', 4.0: '17500', 5.0: '22500', 6.0: '30000', 
              7.0: '40000', 8.0: '50000', 9.0: '60000', 10.0: '70000', 12.0: '30000', 13.0: '10000', 
              14.0: '87500', 15.0: '100000', 
}



diabetes_family_map = {1: 'Yes', 2: 'No', 9: np.nan}

# Apply mappings
df['Gender'] = df['RIAGENDR_imputed'].map(gender_map)
df['Race'] = df['RIDRETH3_imputed'].map(race_map)
df['Education'] = df['DMDEDUC2_imputed'].map(education_map)
df['Income'] = df['INDHHIN2_imputed'].map(income_map)
df['Family_Diabetes'] = df['MCQ300C_imputed'].map(diabetes_family_map)
df['Diabetes_Status'] = df['DIQ010'].map({2.0: 'No Diabetes', 1.0: 'Diabetes'})


# %%

# Convert type
df['Income'] = df['Income'].astype('int64')

# %%


# Create income groups
df['Income_Group'] = np.where(df['INDHHIN2_imputed'].isin([1.0, 2.0, 3.0, 4.0]), 'Low Income',
                      np.where(df['INDHHIN2_imputed'].isin([5.0, 6.0, 7.0, 8.0, 9.0]), 'Medium Income',
                      np.where(df['INDHHIN2_imputed'].isin([10.0, 12.0, 13.0, 14.0, 15.0]), 'High Income', np.nan)))


# %%


# Create risk level based on waist circumference
df['Risk_Level'] = np.where(
    ((df['Gender'] == 'Male') & (df['BMXWAIST_imputed'] >= 102)) | 
    ((df['Gender'] == 'Female') & (df['BMXWAIST_imputed'] >= 88)),
    'High Risk', 'Low Risk'
)


# %%


# Create obesity status categories
df['Obesity_Status'] = pd.cut(
    df['BMXBMI_imputed'],
    bins=[0, 24.9, 29.9, float('inf')],
    labels=['Non-Obese', 'Overweight', 'Obese']
)


# %%


# Remove rows with missing diabetes status (target variable)
initial_count = len(df)
df.dropna(subset='Diabetes_Status', inplace=True)
final_count = len(df)
print(f"Removed {initial_count - final_count} rows with missing diabetes status.")


# %%


# Dataset overview after processing
print("="*60)
print("FINAL PROCESSED DATASET OVERVIEW")
print("="*60)
print(f"Shape: {df.shape}")
print(f"Diabetes prevalence: {df['DIQ010'].mean():.2%}")

print("\nMissing values summary:")
for feature in numeric_features + categorical_features:
    if f'{feature}_missing' in df.columns:
        missing = df[f'{feature}_missing'].sum()
        if missing > 0:
            print(f"{feature}: {missing} missing values")


# %%


# Rename columns to meaningful names
column_mapping = {
    'SEQN': 'ID',
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Gender_Code',
    'RIDRETH3': 'Race_Code',
    'DMDEDUC2': 'Education_Code',
    'INDHHIN2': 'Income_Code',
    'DIQ010': 'Diabetes_Indicator',
    'MCQ300C': 'Family_Diabetes_Code',
    'BMXBMI': 'BMI',
    'BMXWAIST': 'Waist_Circumference',
    'BPXSY1': 'Systolic_BP',
    'BPXDI1': 'Diastolic_BP',
    'LBXGLT': 'Glucose',
    'LBDHDD': 'HDL',
    'LBXTR': 'Triglycerides',
    
    # Missing/imputed versions
    'RIDAGEYR_missing': 'Age_Missing',
    'RIDAGEYR_imputed': 'Age_Imputed',
    'BMXBMI_missing': 'BMI_Missing',
    'BMXBMI_imputed': 'BMI_Imputed',
    'BMXWAIST_missing': 'Waist_Circumference_Missing',
    'BMXWAIST_imputed': 'Waist_Circumference_Imputed',
    'BPXSY1_missing': 'Systolic_BP_Missing',
    'BPXSY1_imputed': 'Systolic_BP_Imputed',
    'BPXDI1_missing': 'Diastolic_BP_Missing',
    'BPXDI1_imputed': 'Diastolic_BP_Imputed',
    'LBXGLT_missing': 'Glucose_Missing',
    'LBXGLT_imputed': 'Glucose_Imputed',
    'LBDHDD_missing': 'HDL_Missing',
    'LBDHDD_imputed': 'HDL_Imputed',
    'LBXTR_missing': 'Triglycerides_Missing',
    'LBXTR_imputed': 'Triglycerides_Imputed',
    'RIAGENDR_missing': 'Gender_Code_Missing',
    'RIAGENDR_imputed': 'Gender_Code_Imputed',
    'RIDRETH3_missing': 'Race_Code_Missing',
    'RIDRETH3_imputed': 'Race_Code_Imputed',
    'DMDEDUC2_missing': 'Education_Code_Missing',
    'DMDEDUC2_imputed': 'Education_Code_Imputed',
    #'INDHHIN2_missing': 'Income_Code_Missing',
    #'INDHHIN2_imputed': 'Income_Code_Imputed',
    'MCQ300C_missing': 'Family_Diabetes_Code_Missing',
    'MCQ300C_imputed': 'Family_Diabetes_Code_Imputed',
    
    # Engineered features
    'Gender': 'Gender',
    'Race': 'Race',
    'Education': 'Education',
    'Income': 'Income',
    'Family_Diabetes': 'Family_Diabetes',
    'Diabetes_Status': 'Diabetes_Status',
    'Income_Numeric': 'Income_Numeric',
    #'Income_Numeric_missing': 'Income_Numeric_Missing',
    #'Income_Numeric_imputed': 'Income_Numeric_Imputed',
    'Income_Group': 'Income_Group',
    'Risk_Level': 'Risk_Level',
    'Obesity_Status': 'Obesity_Status'
}

df = df.rename(columns=column_mapping)


# %%


# Final dataset info
print("Final dataset columns and info:")
df.info()


# %% [markdown]
# ----------------------------------
#                                                              Exploratory Data Analysis
# ----------------------------------

# %%


print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Distribution analysis
print("\n1. Diabetes Status Distribution:")
print(df['Diabetes_Status'].value_counts())

print("\n2. Risk Level Distribution:")
print(df['Risk_Level'].value_counts())

print("\n3. Obesity Status Distribution:")
print(df['Obesity_Status'].value_counts())


# %%


print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Comparative analysis
print("\n4. Health Metrics by Diabetes Status:")
diabetes_comparison = df.groupby('Diabetes_Status')[
    ["BMI_Imputed", "Waist_Circumference_Imputed", "Glucose_Imputed", "Triglycerides_Imputed"]
].mean().round(2)
print(diabetes_comparison)


# %%

print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# BMI Comparison by Gender and Diabetes Status
print("\n5. BMI Comparison by Gender and Diabetes Status:")
print("Males:")
print(df.query("Gender == 'Male'").groupby("Diabetes_Status")["BMI_Imputed"].mean().round(2))

print("\nFemales:")
print(df.query("Gender == 'Female'").groupby("Diabetes_Status")["BMI_Imputed"].mean().round(2))


# %%

print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Income by Obesity Status
print("\n6. Income by Obesity Status:")
print(df.groupby("Obesity_Status")["Income"].mean().round(2))


# %%

print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Income by Education Level (Diabetes Patients)
print("\n7. Income by Education Level (Diabetes Patients):")
income_by_education = df.query("Diabetes_Status == 'Diabetes'").groupby('Education')['Income'].mean()
print(income_by_education.apply(lambda x: f"${x:,.0f}"))

# %% [markdown]
# --------------------------------------
#                                                              DATA VISUALIZATION
# --------------------------------------

# %%


print("\nGenerating visualizations...")

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")

# Figure 1: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age_Imputed'], bins=20, kde=True)
plt.title('Age Distribution (Imputed)', fontsize=14, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%


# Figure 2: BMI Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['BMI_Imputed'], bins=20, kde=True)
plt.title('BMI Distribution (Imputed)', fontsize=14, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%


# Figure 3: Diabetes by Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Gender', hue='Diabetes_Status')
plt.title('Diabetes Prevalence by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Diabetes Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%


# Figure 4: Diabetes by Race
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='Race', hue='Diabetes_Status')
plt.title('Diabetes Prevalence by Race/Ethnicity', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Race/Ethnicity')
plt.legend(title='Diabetes Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%


# Figure 5: Correlation Heatmap
plt.figure(figsize=(12, 10))
imputed_numeric_features = [col for col in df.columns if col.endswith("_Imputed") and col != 'Income_Code_Imputed']
correlation_matrix = df[imputed_numeric_features].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix (Imputed Values)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# %%


# Figure 6: BMI by Diabetes Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_Status', y='BMI_Imputed', data=df)
plt.title('BMI Distribution by Diabetes Status', fontsize=14, fontweight='bold')
plt.xlabel('Diabetes Status')
plt.ylabel('BMI')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# %% [markdown]
# --------------------------------------
#                                                         Save Processed Data
# --------------------------------------

# %%


# Save the processed dataset
output_path = 'nhanes_analysis.csv'
df.to_csv(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")
print(f"Final dataset shape: {df.shape}")


# %% [markdown]
# ---------------------
#                                                     Final Summary Statistics
# ------------------------

# %%
# Final summary statistics
print("\n" + "="*60)
print("FINAL SUMMARY STATISTICS")
print("="*60)

print(f"\nTotal participants: {len(df):,}")
print(f"Diabetes cases: {df[df['Diabetes_Status'] == 'Diabetes'].shape[0]:,}")
print(f"Non-diabetes cases: {df[df['Diabetes_Status'] == 'No Diabetes'].shape[0]:,}")
print(f"Diabetes prevalence: {(df['Diabetes_Status'] == 'Diabetes').mean():.2%}")

print(f"\nHigh risk individuals: {df[df['Risk_Level'] == 'High Risk'].shape[0]:,}")
print(f"Low risk individuals: {df[df['Risk_Level'] == 'Low Risk'].shape[0]:,}")

print(f"\nObesity status:")
print(f"Non-Obese: {df[df['Obesity_Status'] == 'Non-Obese'].shape[0]:,}")
print(f"Overweight: {df[df['Obesity_Status'] == 'Overweight'].shape[0]:,}")
print(f"Obese: {df[df['Obesity_Status'] == 'Obese'].shape[0]:,}")

# %%
# Key health metrics comparison
diabetes_metrics = df.groupby('Diabetes_Status')[
    ['BMI_Imputed', 'Waist_Circumference_Imputed', 'Glucose_Imputed', 'Triglycerides_Imputed']
].mean().round(2)

print(f"\nAverage health metrics by diabetes status:")
print(diabetes_metrics)


# %% [markdown]
# # 🩺 NHANES Diabetes Analysis – Insights & Recommendations
#
# ## 1. Average Health Metrics by Diabetes Status
# | Metric                     | Diabetes | No Diabetes |
# |-----------------------------|----------|-------------|
# | **BMI (kg/m²)**            | 32.18    | 24.91       |
# | **Waist Circumference (cm)** | 109.54   | 84.83       |
# | **Glucose (mg/dL)**        | 139.57   | 103.25      |
# | **Triglycerides (mg/dL)**  | 151.59   | 94.22       |
#
# ✅ People with **diabetes** show **higher BMI, waist circumference, glucose, and triglyceride levels**, all known risk factors.
#

# %% [markdown]
# ## 2. BMI Differences by Gender & Diabetes Status
# - **Males**
#   - Diabetes: **31.20**
#   - No Diabetes: **24.36**
#
# - **Females**
#   - Diabetes: **33.10**
#   - No Diabetes: **25.45**
#
# ⚠️ Diabetic females show **slightly higher BMI** than diabetic males, suggesting stronger obesity-diabetes linkage in women.
#

# %% [markdown]
# ## 3. Obesity & Income Relationship
# | Obesity Status | Avg. Income Code |
# |----------------|------------------|
# | Non-Obese      | 51,703.92             |
# | Overweight     | 53,530.68             |
# | Obese          | 48,522.88             |
#
# 📉 Obese individuals tend to have **slightly lower income levels** compared to non-obese.
#

# %% [markdown]
# ## 4. Education vs Income (Diabetes Patients Only)
# | Education Level             | Avg. Income (USD)  |
# |-----------------------------|--------------------|
# | Less than 9th grade         | 33,137             |
# | 9–11th grade                | 37,198             |
# | High school graduate        | 43,070             |
# | Some college / AA degree    | 48,269             |
# | College graduate            | 71,782             |
#
# 📚 Higher education is linked with **higher income levels** even among people with diabetes.
#

# %% [markdown]
# ## 5. Population Distribution
# - **Diabetes prevalence**
#   - Diabetes: **722**
#   - No Diabetes: **8,514**
#
# - **Risk Levels**
#   - High Risk: **3,382**
#   - Low Risk: **5,854**
#
# - **Obesity Status**
#   - Non-Obese: **4,877**
#   - Overweight: **2,086**
#   - Obese: **2,273**
#
# ⚠️ Most participants are **non-diabetic**, but **over 3,300 individuals are at high risk**, and obesity is strongly present.
#
# ---
#
# # ✅ Key Insights
# 1. **Diabetes is strongly associated** with higher BMI, waist size, glucose, and triglycerides.  
# 2. **Females with diabetes** tend to have higher BMI compared to diabetic males.  
# 3. **Income and education** play protective roles: higher education = higher income = potentially lower diabetes risk.  
# 4. **Obesity is widespread** and strongly overlaps with diabetes and high-risk groups.  
#
# ---
#
# # 💡 Recommendations
# - **Targeted Interventions**: Focus on **obese and overweight individuals**, especially women, for diabetes prevention.  
# - **Education & Awareness**: Promote **health literacy programs** in lower-education groups.  
# - **Socioeconomic Focus**: Consider income-linked strategies, as low-income populations show worse health outcomes.  
# - **Preventive Screenings**: Increase glucose and cholesterol screenings for high-risk groups.  
# - **Lifestyle Programs**: Encourage **weight management, physical activity, and balanced diets** to reduce risks.  
# 👉 This way, your notebook will have both tables + narrative insights + recommendations in Markdown, without clutter.
#
#
#

# %% [markdown]
# --------------------------------
#                                                             Project Information
# --------------------------------

# %%



print("\n" + "="*60)
print("PROJECT INFORMATION")
print("="*60)
print("Project: NHANES Diabetes Predictive Analysis")
print("Description: Comprehensive analysis of diabetes risk factors using NHANES data")
print("Data Source: National Health and Nutrition Examination Survey")
print("Author: Rotimi Sheriff Omosewo")
print("Contact: omoseworotimi@gmail.com")
print("GitHub: https://github.com/rotimi2020")
print("LinkedIn: https://www.linkedin.com/in/rotimi-sheriff-omosewo-939a806b/")

# %%

# %%

# %%

# %%
