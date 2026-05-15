import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page config
st.set_page_config(page_title="Engine ML Dashboard", layout="wide")

# Light blue background CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #e6f0ff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Engine Failure ML Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("📄 Upload the Engine Failure Dataset", type=["csv"])

if uploaded_file:
    # Read and clean data
    df = pd.read_csv(uploaded_file, encoding='latin1')
    df.rename(columns={"Temperature (Â°C)": "Temperature (°C)"}, inplace=True)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())

    # Features and target
    features = ['RPM', 'Torque', 'Fuel_Efficiency', 'Power_Output (kW)',
                'Vibration_X', 'Vibration_Y', 'Vibration_Z']
    X = df[features]
    y = df['Temperature (°C)']

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model selection
    st.subheader("🔍 Choose a Regression Model")
    model_option = st.selectbox("Select a model", 
                                ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "KNN Regression"])

    # Initialize model
    if model_option == "Linear Regression":
        model = LinearRegression()
        color = 'green'
    elif model_option == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=6, random_state=0)
        color = 'orange'
    elif model_option == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        color = 'teal'
    elif model_option == "SVR":
        model = SVR(kernel='rbf')
        color = 'purple'
    elif model_option == "KNN Regression":
        model = KNeighborsRegressor(n_neighbors=5)
        color = 'blue'

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader(f"📌 {model_option}")
    st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

    # Feature importance for tree-based models
    if model_option in ["Decision Tree", "Random Forest"]:
        st.subheader("📊 Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig, ax = plt.subplots()
        importance.sort_values().plot(kind='barh', color=color, ax=ax)
        st.pyplot(fig)

    # User input prediction
    st.markdown("---")
    st.subheader("🎯 Predict Output Based on Your Input")
    st.markdown("Enter custom values:")

    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(feature, value=float(df[feature].mean()), format="%.2f")

    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    user_prediction = model.predict(user_scaled)[0]

    st.markdown("### 🧪 Prediction Result")
    st.success(f"Predicted Temperature (°C): {round(user_prediction, 2)}")

    st.markdown("### 📊 Prediction Graph")
    fig_bar, ax_bar = plt.subplots(figsize=(4, 2))
    ax_bar.barh(["Temperature (°C)"], [user_prediction], color=color)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_title("Predicted Temperature")
    st.pyplot(fig_bar)

    # Comparison
    st.markdown("---")
    st.subheader("📊 Compare All Regression Models")

    col1, col2 = st.columns(2)
    compare_yes = col1.button("Yes ✅")
    compare_no = col2.button("No ❌")

    if compare_yes:
        all_models = []
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
            "SVR": SVR(kernel='rbf'),
            "KNN Regression": KNeighborsRegressor(n_neighbors=5)
        }

        for name, m in models.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            all_models.append([name, mae, rmse])

        result_df = pd.DataFrame(all_models, columns=["Model", "MAE", "RMSE"])
        best_model = result_df.sort_values(by="MAE", ascending=True).iloc[0]["Model"]

        st.dataframe(result_df)

        st.markdown("### 📈 Model Comparison (MAE)")
        fig_cmp, ax_cmp = plt.subplots()
        sns.barplot(data=result_df, x="MAE", y="Model", palette="coolwarm", ax=ax_cmp)
        st.pyplot(fig_cmp)

        st.success(f"🏆 Best Performing Model: {best_model}")

else:
    st.info("📅 Please upload your engine_failure_dataset.csv to begin.")
