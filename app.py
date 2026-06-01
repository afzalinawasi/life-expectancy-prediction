import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load saved model and preprocessing objects
model = joblib.load("artifacts/final_random_forest_model.pkl")
preprocessing_objects = joblib.load("artifacts/preprocessing_objects.pkl")

scaler = preprocessing_objects["scaler"]
train_medians = preprocessing_objects["train_medians"]
numeric_cols = preprocessing_objects["numeric_cols"]
feature_columns = preprocessing_objects["feature_columns"]
status_mapping = preprocessing_objects["status_mapping"]


st.set_page_config(
    page_title="Life expectancy prediction",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for cleaner UI
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }

    h1, h2, h3 {
        color: #2b2d35;
    }

    .intro-card {
        background-color: #f5f5f5;
        padding: 1.2rem 1.4rem;
        border-left: 6px solid #d71920;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background-color: #111111;
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border-bottom: 4px solid #d71920;
    }

    .section-divider {
        border-top: 1px solid #dddddd;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Prediction app", "Life expectancy prediction insights"]
)


# Banner
st.image("assets/Life expectancy banner.png", use_container_width=True)

if page == "Prediction app":
    st.markdown(
        """
        <div class="intro-card">
            <h2>Life expectancy prediction using machine learning</h2>
            <p>
            This app predicts life expectancy using country-level health, economic,
            demographic, and education-related indicators. The final selected model is
            a <b>Random Forest Regressor</b>, chosen after model comparison and cross-validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.markdown(
        """
        <div class="intro-card">
            <h2>Life expectancy prediction insights dashboard</h2>
            <p>
            This dashboard explains the key drivers behind the life expectancy prediction model,
            compares model performance, and shows how selected input values compare with dataset
            benchmarks. It is designed to help users understand the model outcome, not just
            generate a prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Insights page
if page == "Life expectancy prediction insights":

    st.markdown("### Life expectancy prediction insights")
    st.caption(
        "These charts explain the model behaviour, model selection, and how the selected input profile compares with dataset benchmarks."
    )

    def insight_card(title, what_it_says, key_takeaway):
        st.markdown(
            f"""
            <div style="
                background-color:#f5f5f5;
                padding:1.2rem 1.4rem;
                border-left:6px solid #d71920;
                border-radius:10px;
                margin-top:0.8rem;
                margin-bottom:1.6rem;
            ">
                <h4 style="margin-bottom:0.5rem; color:#2b2d35;">{title}</h4>
                <p style="margin-bottom:0.4rem; color:#333333;">
                    <b>What the chart says:</b> {what_it_says}
                </p>
                <p style="margin-bottom:0rem; color:#333333;">
                    <b>Key takeaway:</b> {key_takeaway}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Load dataset for benchmark chart
    dataset_df = pd.read_csv("Life Expectancy Data.csv")
    dataset_df.columns = dataset_df.columns.str.strip()

    # Chart 1: Feature importance
    st.markdown("#### 1. Top 10 feature importance")

    feature_importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    top_features = feature_importance_df.head(10).sort_values(by="Importance")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_features["Feature"], top_features["Importance"], color="#d71920")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 feature importance from Random Forest")
    plt.tight_layout()

    st.pyplot(fig)

    insight_card(
        "Feature importance insight",
        "The model relies most strongly on HIV/AIDS, income composition of resources, and adult mortality while predicting life expectancy.",
        "Disease burden, resource availability, and mortality risk are the most important model drivers. The developed/developing label alone does not drive the prediction."
    )

    # Chart 2: Model performance comparison
    st.markdown("#### 2. Model performance comparison")

    model_performance_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "KNN",
            "XGBoost"
        ],
        "Test R2": [
            0.819488,
            0.920800,
            0.966979,
            0.733839,
            0.855567,
            0.965664
        ],
        "Test RMSE": [
            3.951483,
            2.617394,
            1.690046,
            4.798208,
            3.534598,
            1.723380
        ],
        "Test MAE": [
            2.927327,
            1.539590,
            1.055229,
            3.592166,
            2.488771,
            1.148376
        ]
    }).sort_values(by="Test R2", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(model_performance_df["Model"], model_performance_df["Test R2"], color="#d71920")
    ax.set_xlabel("Test R²")
    ax.set_ylabel("Model")
    ax.set_title("Model comparison by Test R²")
    ax.set_xlim(0, 1)
    plt.tight_layout()

    st.pyplot(fig)

    st.dataframe(
        model_performance_df.sort_values(by="Test R2", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    insight_card(
        "Model comparison insight",
        "Random Forest achieved the highest Test R² and the lowest Test RMSE and Test MAE among the compared models.",
        "Random Forest was selected because it gave the best balance of accuracy and generalization, with XGBoost performing very close behind."
    )

    # Chart 3: Input profile vs dataset benchmark
    st.markdown("#### 3. Input profile vs dataset benchmark")

    default_input_data = {
        "Year": 2015,
        "Status": "Developing",
        "Adult Mortality": 150.0,
        "infant deaths": 10.0,
        "Alcohol": 4.0,
        "percentage expenditure": 100.0,
        "Hepatitis B": 80.0,
        "Measles": 0.0,
        "BMI": 40.0,
        "under-five deaths": 10.0,
        "Polio": 85.0,
        "Total expenditure": 6.0,
        "Diphtheria": 85.0,
        "HIV/AIDS": 0.1,
        "GDP": 3000.0,
        "Population": 1000000,
        "thinness  1-19 years": 5.0,
        "thinness 5-9 years": 5.0,
        "Income composition of resources": 0.70,
        "Schooling": 12.0
    }

    selected_profile = st.session_state.get("latest_input_data", default_input_data)
    selected_status = selected_profile["Status"]

    benchmark_df = dataset_df[dataset_df["Status"] == selected_status]

    benchmark_features = [
        "HIV/AIDS",
        "Adult Mortality",
        "Income composition of resources",
        "Schooling"
    ]

    benchmark_rows = []

    for feature in benchmark_features:
        selected_value = float(selected_profile[feature])
        benchmark_value = float(benchmark_df[feature].median())

        if benchmark_value != 0:
            percent_difference = ((selected_value - benchmark_value) / benchmark_value) * 100
        else:
            percent_difference = 0

        benchmark_rows.append({
            "Indicator": feature,
            "Selected value": selected_value,
            "Dataset median": benchmark_value,
            "Difference from median (%)": percent_difference
        })

    benchmark_comparison_df = pd.DataFrame(benchmark_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        benchmark_comparison_df["Indicator"],
        benchmark_comparison_df["Difference from median (%)"],
        color="#d71920"
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Difference from dataset median (%)")
    ax.set_ylabel("Indicator")
    ax.set_title(f"Selected input profile vs {selected_status} country benchmark")
    plt.tight_layout()

    st.pyplot(fig)

    st.dataframe(
        benchmark_comparison_df,
        use_container_width=True,
        hide_index=True
    )

    insight_card(
        "Input profile benchmark insight",
        "The chart compares the selected profile against the median values for the same development-status group in the dataset.",
        "Positive values are not always good. Higher income composition and schooling are favourable, while higher HIV/AIDS and adult mortality indicate higher risk."
    )

    st.stop()

st.markdown("### Enter prediction inputs")
st.caption("Adjust the values below. Inputs are grouped by feature type.")

profile_tab, economic_tab = st.tabs(
    ["Health and profile indicators", "Economic and social indicators"]
)

with profile_tab:
    st.markdown("#### Basic, health, mortality, and immunization details")

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.slider("Year", min_value=2000, max_value=2015, value=2015)
        status = st.selectbox("Development status", ["Developing", "Developed"])
        adult_mortality = st.number_input(
            "Adult Mortality",
            min_value=0.0,
            max_value=800.0,
            value=150.0,
            step=1.0,
            format="%.2f"
        )
        infant_deaths = st.number_input("Infant deaths", min_value=0.0, value=10.0, step=1.0)

    with col2:
        under_five_deaths = st.number_input("Under-five deaths", min_value=0.0, value=10.0, step=1.0)
        hiv_aids = st.slider("HIV/AIDS", min_value=0.0, max_value=50.0, value=0.1, step=0.1)
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=90.0,
            value=40.0,
            step=0.1,
            format="%.2f"
        )
        alcohol = st.number_input(
            "Alcohol",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.1,
            format="%.2f"
        )

    with col3:
        measles = st.number_input("Measles", min_value=0.0, value=0.0, step=1.0)
        hepatitis_b = st.slider("Hepatitis B", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        polio = st.slider("Polio", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        diphtheria = st.slider("Diphtheria", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

with economic_tab:
    st.markdown("#### Economic, population, education, and social indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        gdp = st.number_input("GDP", min_value=0.0, value=3000.0, step=100.0)
        percentage_expenditure = st.number_input("Percentage expenditure", min_value=0.0, value=100.0, step=10.0)

    with col2:
        population = st.number_input(
            "Population",
            min_value=0,
            value=1000000,
            step=10000,
            format="%d"
        )
        total_expenditure = st.slider("Total expenditure", min_value=0.0, max_value=20.0, value=6.0, step=0.1)

    with col3:
        income_composition = st.number_input(
            "Income composition of resources",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.01,
            format="%.2f"
        )
        schooling = st.slider("Schooling", min_value=0.0, max_value=25.0, value=12.0, step=0.1)
        thinness_1_19 = st.slider("Thinness 1-19 years", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
        thinness_5_9 = st.slider("Thinness 5-9 years", min_value=0.0, max_value=30.0, value=5.0, step=0.1)


# Prepare input data
input_data = {
    "Year": year,
    "Status": status,
    "Adult Mortality": adult_mortality,
    "infant deaths": infant_deaths,
    "Alcohol": alcohol,
    "percentage expenditure": percentage_expenditure,
    "Hepatitis B": hepatitis_b,
    "Measles": measles,
    "BMI": bmi,
    "under-five deaths": under_five_deaths,
    "Polio": polio,
    "Total expenditure": total_expenditure,
    "Diphtheria": diphtheria,
    "HIV/AIDS": hiv_aids,
    "GDP": gdp,
    "Population": population,
    "thinness  1-19 years": thinness_1_19,
    "thinness 5-9 years": thinness_5_9,
    "Income composition of resources": income_composition,
    "Schooling": schooling
}

# Store latest selected inputs for the insights page
st.session_state["latest_input_data"] = input_data.copy()

input_df = pd.DataFrame([input_data])

# Show input preview before prediction
preview_df = input_df[feature_columns].copy()

preview_display = preview_df.T.reset_index()
preview_display.columns = ["Indicator", "Selected value"]

st.markdown("### Input preview")
st.caption("Review the selected values before running the prediction.")

st.dataframe(
    preview_display,
    use_container_width=True,
    hide_index=True
)

# Keep same feature order as training
input_df = input_df[feature_columns]

# Apply same preprocessing as notebook
input_df["Status"] = input_df["Status"].map(status_mapping)


input_df[numeric_cols] = input_df[numeric_cols].fillna(train_medians)
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

if st.button("Predict life expectancy"):
    prediction = float(model.predict(input_df)[0])

    # Dataset median benchmarks from EDA
    if status == "Developing":
        benchmark_median = 69.0
        benchmark_group = "developing-country"
    else:
        benchmark_median = 79.25
        benchmark_group = "developed-country"

    difference_from_benchmark = prediction - benchmark_median

    if difference_from_benchmark > 2:
        benchmark_message = (
            f"This prediction is around {difference_from_benchmark:.1f} years above "
            f"the typical {benchmark_group} median in the dataset."
        )
    elif difference_from_benchmark < -2:
        benchmark_message = (
            f"This prediction is around {abs(difference_from_benchmark):.1f} years below "
            f"the typical {benchmark_group} median in the dataset."
        )
    else:
        benchmark_message = (
            f"This prediction is close to the typical {benchmark_group} median in the dataset."
        )

    if prediction >= 75:
        range_label = "High life expectancy range"
        range_message = (
            "The profile shows stronger health, income, education, and immunization indicators."
        )
        result_color = "#e8f5e9"
        border_color = "#1b8f3a"

    elif prediction >= 65:
        range_label = "Medium-to-high life expectancy range"
        range_message = (
            "The profile shows reasonable overall indicators, but mortality, disease burden, "
            "or economic conditions may still limit the predicted life expectancy."
        )
        result_color = "#fff8e1"
        border_color = "#d18b00"

    else:
        range_label = "Low life expectancy range"
        range_message = (
            "The profile may reflect higher mortality, disease burden, weaker economic indicators, "
            "or lower social development conditions."
        )
        result_color = "#fdecea"
        border_color = "#d71920"

    st.markdown("### Prediction result")

    st.markdown(
        f"""
        <div style="
            background-color:{result_color};
            border-left:7px solid {border_color};
            padding:1.3rem 1.5rem;
            border-radius:12px;
            margin-top:1rem;
            margin-bottom:1rem;
        ">
            <h2 style="margin-bottom:0.3rem; color:#2b2d35;">
                Predicted life expectancy: {prediction:.2f} years
            </h2>
            <h4 style="margin-top:0.5rem; color:#2b2d35;">
                {range_label}
            </h4>
            <p style="font-size:1rem; color:#333333;">
                {benchmark_message}
            </p>
            <p style="font-size:1rem; color:#333333;">
                {range_message}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Model context: In this project, the strongest prediction drivers were HIV/AIDS, "
        "income composition of resources, and adult mortality. This output is a machine learning "
        "estimate based on the selected inputs, not a medical or policy conclusion."
    )

st.markdown(
    """
    ---
    **Model:** Random Forest Regressor  
    **Dataset:** [WHO Life Expectancy dataset on Kaggle](https://www.kaggle.com/datasets/vikramamin/life-expectancy-who/data)
    """
)