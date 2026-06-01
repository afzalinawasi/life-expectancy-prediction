# Life expectancy prediction using machine learning

This project predicts life expectancy using country-level health, demographic, immunization, economic, and education indicators from the WHO Life Expectancy dataset.

The final model is a **Random Forest Regressor**, selected after model comparison, overfitting checks, and cross-validation.

---

## Project objective

The objective is to build a supervised regression model that answers:

> What is the expected life expectancy based on country-level health and socio-economic indicators?

The project also includes a Streamlit app where users can enter indicator values, generate a prediction, and view simple model insights.

---

## Dataset

Dataset: [WHO Life Expectancy dataset on Kaggle](https://www.kaggle.com/datasets/vikramamin/life-expectancy-who/data)

The dataset contains:

- 2,938 rows
- 22 columns
- Country-year level records
- Health, mortality, immunization, economic, population, and education indicators

Target variable:

```
Life expectancy
```

Since the target is a continuous numerical value, this is a supervised regression problem.

---
## Assignment coverage

| Requirement                 | Covered in project                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Load dataset                | Loaded using Pandas                                                                                                      |
| EDA and visualizations      | Checked data structure, missing values, distributions, skewness, outliers, correlations, and status-wise life expectancy |
| Distribution/skewness check | Checked target and numerical feature skewness                                                                            |
| Transformation decision     | Considered but not applied blindly; explanation added                                                                    |
| Outlier treatment           | Checked using IQR and boxplots; retained valid country-year outliers                                                     |
| Feature scaling             | Applied RobustScaler after train-test split                                                                              |
| Model building              | Trained multiple supervised regression models                                                                            |
| Overfitting check           | Compared train and test performance                                                                                      |
| Model comparison            | Compared models using MAE, RMSE, and R²                                                                                  |
| Final model selection       | Selected Random Forest based on test and cross-validation performance                                                    |
| Deployment                  | Built a Streamlit prediction app and insights dashboard                                                                  |

---

## Data cleaning and preprocessing

The following preprocessing steps were performed:

Cleaned column names by removing extra spaces
Removed rows where the target value was missing
Dropped Country to avoid high-cardinality encoding
Encoded Status as:
Developing = 0
Developed = 1
Split data into training and testing sets
Filled missing numerical values using training-data medians
Applied RobustScaler to reduce the impact of outliers
Saved preprocessing objects for Streamlit app usage

---

## Exploratory data analysis summary

Key EDA findings:
- The dataset has missing values in both the target and several input columns.
- The target column Life expectancy has 10 missing values.
- Life expectancy ranges from 36.3 to 89.0 years.
- The target is moderately left-skewed.
- Several numerical features such as Population, Measles, infant deaths, under-five deaths, HIV/AIDS, and GDP are highly skewed.
- Outliers are present, but they may represent valid country-year observations.
- Schooling and Income composition of resources show strong positive correlation with life expectancy.
- Adult Mortality and HIV/AIDS show strong negative correlation with life expectancy.
- Developed countries show higher median life expectancy than developing countries.
  
---
## Model training

The following supervised regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- KNN Regressor
- XGBoost Regressor

The models were evaluated using:
- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score

----
## Model evaluation comparison

| Model             | Test R² | Test RMSE | Test MAE |
| ----------------- | ------: | --------: | -------: |
| Random Forest     |  0.9670 |    1.6900 |   1.0552 |
| XGBoost           |  0.9656 |    1.7238 |   1.1484 |
| Decision Tree     |  0.9208 |    2.6179 |   1.5399 |
| KNN               |  0.8557 |    3.5346 |   2.4888 |
| Linear Regression |  0.8195 |    3.9515 |   2.9273 |
| SVM               |  0.7338 |    4.7982 |   3.5922 |


---
## Final model selected

**Random Forest Regressor** was selected as the final model.

Final model performance:
- Test R²: 0.9669
- Test RMSE: 1.6900
- Test MAE: 1.0552
- Cross-validation R² mean: 0.9567

Random Forest was selected because it gave the best overall balance of test accuracy, low error, and cross-validation stability.

---
## Streamlit app features

The Streamlit app allows users to:

Enter health, mortality, immunization, economic, and social indicators
Preview selected input values
Predict life expectancy
View a simple prediction range interpretation
Compare the prediction with development-status benchmarks

The app also includes a short model context note explaining that the prediction is a machine learning estimate, not a medical or policy conclusion.

Link to view the app: 

---

## Life expectancy insights dashboard

The app includes an insights dashboard with:

- Top 10 feature importance
- Shows the strongest model drivers behind the prediction.
- Model performance comparison
- Compares supervised ML models using test performance.
- Input profile vs dataset benchmark
- Compares selected inputs with dataset median values for the selected development-status group.

---

## Tech stack

The following concepts were used: 
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---
## How to run the project

After activating myenv and validating the Python version as 3.12, follow these steps:

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run all cells in code.ipynb file 

3. Run the Streamlit app:

```
streamlit run app.py
```

---
## Limitation

This is a learning project based on historical country-level data. The prediction should be interpreted as a machine learning estimate and not as a medical, demographic, or policy conclusion.
