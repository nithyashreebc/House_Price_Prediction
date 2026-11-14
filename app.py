
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="HomePredict",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# 2. LOAD DATASET AND MODELS
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        BASE_DIR / "data" / "cleaned_house_data.csv"
    )


@st.cache_resource
def load_models():

    linear_model = joblib.load(
        BASE_DIR / "models" / "linear_regression_model.pkl"
    )

    rf_model = joblib.load(
        BASE_DIR / "models" / "random_forest_model.pkl"
    )

    return linear_model, rf_model


df = load_data()
linear_model, rf_model = load_models()

# =====================================================
# 3. DATABASE CONFIGURATION
# =====================================================

DB_PATH = BASE_DIR / "prediction_history.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_history_table():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            location TEXT,
            area REAL,
            bhk INTEGER,
            bathrooms INTEGER,
            model TEXT,
            predicted_price REAL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    location,
    area,
    bhk,
    bathrooms,
    model,
    price
):

    conn = get_connection()

    conn.execute("""
        INSERT INTO prediction_history
        (
            prediction_date,
            location,
            area,
            bhk,
            bathrooms,
            model,
            predicted_price
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%d-%m-%Y %H:%M"),
        location,
        area,
        bhk,
        bathrooms,
        model,
        price
    ))

    conn.commit()
    conn.close()


def load_prediction_history():

    conn = get_connection()

    history = pd.read_sql_query(
        "SELECT * FROM prediction_history ORDER BY id DESC",
        conn
    )

    conn.close()

    return history


def clear_prediction_history():

    conn = get_connection()

    conn.execute("DELETE FROM prediction_history")

    conn.commit()
    conn.close()


create_history_table()

# =====================================================
# 4. CUSTOM CSS DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f3ff;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #211747,
        #49358c
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main title */

.main-title {
    color: #49358c;
    font-size: 34px;
    font-weight: 800;
}

.sub-title {
    color: #6d5a9e;
    font-size: 16px;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #7354c7;
    box-shadow: 0 3px 12px rgba(70, 50, 120, 0.10);
}

/* Result card */

.result-card {
    background: linear-gradient(
        120deg,
        #49358c,
        #7960ce
    );
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

/* Section headings */

.section-title {
    color: #49358c;
    font-size: 23px;
    font-weight: 700;
}

/* Buttons */

div.stButton > button {
    background-color: #6346b5;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 9px 20px;
}

div.stButton > button:hover {
    background-color: #49358c;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 🏠 HomePredict")

st.sidebar.caption(
    "House Price Prediction System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Predict Price",
        "Prediction History",
        "Visual Analytics",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Machine Learning Project"
)

# =====================================================
# 6. HELPER FUNCTIONS
# =====================================================

def show_metric(label, value):

    st.metric(
        label=label,
        value=value
    )


def draw_horizontal_chart(data, title, xlabel):

    fig, ax = plt.subplots(figsize=(5, 2.3))

    data = data.sort_values()

    ax.barh(
        data.index.astype(str),
        data.values,
        color="#7354c7",
        height=0.35
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        xlabel,
        fontsize=9
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# =====================================================
# 7. OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">HomePredict Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Bengaluru House Price Prediction Using Machine Learning</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        show_metric(
            "Locations",
            df["location"].nunique()
        )

    with col3:
        show_metric(
            "Average Price",
            f"₹{df['price'].mean():.2f} L"
        )

    with col4:
        show_metric(
            "Average Area",
            f"{df['total_sqft'].mean():.0f} sqft"
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        "HomePredict is a machine learning application that estimates "
        "Bengaluru house prices using location, area, bedrooms and bathrooms."
    )

    col1, col2 = st.columns(2)

    with col1:

        location_prices = (
            df.groupby("location")["price"]
            .mean()
            .nlargest(10)
        )

        draw_horizontal_chart(
            location_prices,
            "Top 10 Locations by Average Price",
            "Average Price (Lakhs)"
        )

    with col2:

        bhk_counts = (
            df["bhk"]
            .value_counts()
            .sort_index()
        )

        draw_horizontal_chart(
            bhk_counts,
            "Properties by BHK",
            "Number of Properties"
        )


# =====================================================
# 8. PREDICT PRICE PAGE
# =====================================================

elif page == "Predict Price":

    st.markdown(
        '<div class="main-title">House Price Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the property details to estimate its price."
    )

    locations = sorted(
        df["location"].dropna().unique()
    )

    col1, col2 = st.columns(2)

    with col1:

        location = st.selectbox(
            "Select Location",
            locations
        )

        area = st.number_input(
            "Total Area (sq.ft.)",
            min_value=300.0,
            max_value=50000.0,
            value=1200.0,
            step=100.0
        )

    with col2:

        bhk = st.number_input(
            "Number of Bedrooms (BHK)",
            min_value=1,
            max_value=10,
            value=2
        )

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=12,
            value=2
        )

    model_choice = st.selectbox(
        "Choose Machine Learning Model",
        [
            "Random Forest",
            "Linear Regression"
        ]
    )

    if st.button(
        "Predict House Price",
        use_container_width=True
    ):

        if area / bhk < 300:

            st.error(
                "Please enter at least 300 sq.ft. per BHK."
            )

        elif bathrooms > bhk + 2:

            st.error(
                "Please enter a valid number of bathrooms."
            )

        else:

            input_data = pd.DataFrame([{
                "location": location,
                "total_sqft": area,
                "bath": bathrooms,
                "bhk": bhk
            }])

            if model_choice == "Random Forest":

                prediction = rf_model.predict(
                    input_data
                )[0]

            else:

                prediction = linear_model.predict(
                    input_data
                )[0]

            if prediction <= 0:

                st.error(
                    "The model returned an invalid prediction."
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3>Estimated House Price</h3>
                        <h1>₹ {prediction:.2f} Lakhs</h1>
                        <p>
                            {location} |
                            {bhk} BHK |
                            {area:.0f} sq.ft.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                save_prediction(
                    location,
                    area,
                    bhk,
                    bathrooms,
                    model_choice,
                    float(prediction)
                )

                st.success(
                    "Prediction saved successfully!"
                )

                st.caption(
                    "Note: This is an estimate based on housing "
                    "listing data, not a formal property valuation."
                )


# =====================================================
# 9. PREDICTION HISTORY PAGE
# =====================================================

elif page == "Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    history_df = load_prediction_history()

    if history_df.empty:

        st.info(
            "No prediction history available yet."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            show_metric(
                "Total Predictions",
                len(history_df)
            )

        with col2:

            show_metric(
                "Average Predicted Price",
                f"₹{history_df['predicted_price'].mean():.2f} L"
            )

        st.markdown("---")

        st.subheader("Saved Predictions")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="HomePredict_History.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Prediction History"):

            clear_prediction_history()

            st.success(
                "Prediction history cleared."
            )

            st.rerun()


# =====================================================
# 10. VISUAL ANALYTICS PAGE
# =====================================================

elif page == "Visual Analytics":

    st.markdown(
        '<div class="main-title">Visual Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore your saved house price predictions through charts."
    )

    analytics_df = load_prediction_history()

    if analytics_df.empty:

        st.info(
            "Make some predictions first to view visual analytics."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            show_metric(
                "Total Predictions",
                len(analytics_df)
            )

        with col2:

            show_metric(
                "Average Predicted Price",
                f"₹{analytics_df['predicted_price'].mean():.2f} L"
            )

        with col3:

            show_metric(
                "Locations Analysed",
                analytics_df["location"].nunique()
            )

        st.markdown("---")

        location_counts = (
            analytics_df["location"]
            .value_counts()
            .head(10)
        )

        location_prices = (
            analytics_df.groupby("location")["predicted_price"]
            .mean()
            .nlargest(10)
        )

        model_counts = (
            analytics_df["model"]
            .value_counts()
        )

        model_prices = (
            analytics_df.groupby("model")["predicted_price"]
            .mean()
        )

        # First row of charts

        col1, col2 = st.columns(2)

        with col1:

            draw_horizontal_chart(
                location_counts,
                "Predictions by Location",
                "Number of Predictions"
            )

        with col2:

            draw_horizontal_chart(
                location_prices,
                "Average Predicted Price by Location",
                "Price (Lakhs)"
            )

        # Second row of charts

        col1, col2 = st.columns(2)

        with col1:

            draw_horizontal_chart(
                model_counts,
                "Model Usage",
                "Number of Predictions"
            )

        with col2:

            draw_horizontal_chart(
                model_prices,
                "Average Price by Model",
                "Price (Lakhs)"
            )

        st.markdown("---")

        st.subheader("Analytics Data")

        st.dataframe(
            analytics_df,
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 11. MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    performance = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Random Forest"
        ],

        "MAE": [
            42.43,
            32.77
        ],

        "RMSE": [
            81.00,
            85.17
        ],

        "R2 Score": [
            0.5715,
            0.5263
        ]

    })

    st.subheader("Model Comparison")

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(5, 3))

        ax.bar(
            performance["Model"],
            performance["MAE"],
            color=["#9b82dc", "#6346b5"]
        )

        ax.set_title("Mean Absolute Error (MAE)")
        ax.set_ylabel("Error in Lakhs")

        ax.tick_params(
            axis="x",
            rotation=10
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        fig, ax = plt.subplots(figsize=(5, 3))

        ax.bar(
            performance["Model"],
            performance["RMSE"],
            color=["#9b82dc", "#6346b5"]
        )

        ax.set_title("Root Mean Squared Error (RMSE)")
        ax.set_ylabel("Error in Lakhs")

        ax.tick_params(
            axis="x",
            rotation=10
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    st.info(
        "Random Forest has a lower MAE, while Linear Regression "
        "has a lower RMSE and higher R² score. Neither model "
        "performs best on every evaluation metric."
    )


# =====================================================
# 12. ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">About HomePredict</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Bengaluru House Price Prediction Using Machine Learning"
    )

    st.write(
        "HomePredict is a machine learning application developed "
        "to estimate house prices in Bengaluru based on property features."
    )

    st.markdown("### Project Objective")

    st.write(
        "To estimate house prices using location, total area, "
        "number of bedrooms and bathrooms."
    )

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib
    - Seaborn
    - SQLite
    """)

    st.markdown("### Machine Learning Algorithms")

    st.write("""
    - Linear Regression
    - Random Forest Regressor
    """)

    st.markdown("### Dataset Information")

    st.write("""
    The project uses Bengaluru housing listing data containing:

    - Location
    - Total Square Feet
    - Number of Bedrooms (BHK)
    - Number of Bathrooms
    - House Price
    """)

    st.markdown("### Main Features")

    st.write("""
    - House Price Prediction
    - Multiple Machine Learning Models
    - Prediction History
    - Visual Analytics Dashboard
    - Model Performance Comparison
    - CSV Report Download
    """)


# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <center>
        🏠 HomePredict |
        House Price Prediction Using Machine Learning
    </center>
    """,
    unsafe_allow_html=True
)
