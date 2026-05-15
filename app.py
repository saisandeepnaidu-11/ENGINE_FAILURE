import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Engine Failure ML Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Engine Failure ML Dashboard")
st.markdown("Predictive maintenance and failure analysis")

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

# Load dataset
@st.cache_data
def load_data():
    try:
        # Try multiple paths
        paths = [
            "engine_failure_dataset.csv",
            r"c:\Users\SAI SANDEEP NAIDU\OneDrive\Desktop\ENGINE_FAILLURE\engine_failure_dataset.csv"
        ]
        
        df = None
        for path in paths:
            try:
                df = pd.read_csv(path)
                break
            except:
                continue
        
        if df is None:
            st.error("Dataset not found. Please ensure 'engine_failure_dataset.csv' is in the project directory.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Display dataset info
    st.sidebar.subheader("Dataset Overview")
    st.sidebar.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🔍 Analysis", "🤖 Models", "📈 Predictions"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Data Visualization")
        
        # Select columns for visualization
        if len(df.columns) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numerical Features Distribution**")
                # Plot distribution for numerical columns
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    selected_col = st.selectbox("Select column:", num_cols, key="dist_col")
                    fig, ax = plt.subplots()
                    ax.hist(df[selected_col].dropna(), bins=30, color='steelblue', edgecolor='black')
                    ax.set_title(f"Distribution of {selected_col}")
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
            
            with col2:
                st.write("**Correlation Matrix**")
                num_df = df.select_dtypes(include=[np.number])
                if len(num_df.columns) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
    
    with tab3:
        st.subheader("Machine Learning Models")
        
        # Prepare data
        num_df = df.select_dtypes(include=[np.number])
        
        if len(num_df.columns) > 1:
            # Select target variable
            target_col = st.selectbox("Select target variable:", num_df.columns)
            
            # Drop target from features
            X = num_df.drop(columns=[target_col])
            y = num_df[target_col]
            
            # Split data
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.write(f"**Training set size:** {len(X_train)} | **Test set size:** {len(X_test)}")
            
            # Models to train
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf'),
                'KNN': KNeighborsRegressor(n_neighbors=5)
            }
            
            # Train and evaluate models
            results = {}
            col1, col2, col3 = st.columns(3)
            
            with st.spinner("Training models..."):
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'model': model
                    }
            
            st.success("Models trained successfully!")
            
            # Display results
            st.subheader("Model Performance Comparison")
            
            results_df = pd.DataFrame({
                'Model': list(results.keys()),
                'RMSE': [results[m]['RMSE'] for m in results.keys()],
                'MAE': [results[m]['MAE'] for m in results.keys()],
                'R² Score': [results[m]['R2'] for m in results.keys()]
            })
            
            st.dataframe(results_df.sort_values('R² Score', ascending=False), use_container_width=True)
            
            # Visualize results
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            results_df.set_index('Model').plot(ax=axes[0], kind='bar')
            axes[0].set_title('Model Performance Metrics')
            axes[0].set_ylabel('Score')
            axes[0].legend(loc='best')
            
            axes[1].bar(results_df['Model'], results_df['R² Score'], color='steelblue')
            axes[1].set_title('R² Score Comparison')
            axes[1].set_ylabel('R² Score')
            axes[1].tick_params(axis='x', rotation=45)
            
            axes[2].bar(results_df['Model'], results_df['RMSE'], color='coral')
            axes[2].set_title('RMSE Comparison')
            axes[2].set_ylabel('RMSE')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Make Predictions")
        st.info("Use the sliders below to input feature values and get predictions from all models")
        
        num_df = df.select_dtypes(include=[np.number])
        if len(num_df.columns) > 1:
            target_col = st.selectbox("Select target variable:", num_df.columns, key="pred_target")
            X = num_df.drop(columns=[target_col])
            y = num_df[target_col]
            
            # Train final model for predictions
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Get user input
            st.write("**Input Features:**")
            input_dict = {}
            
            cols = st.columns(2)
            for idx, col in enumerate(X.columns):
                with cols[idx % 2]:
                    min_val = X[col].min()
                    max_val = X[col].max()
                    input_dict[col] = st.slider(
                        f"{col}",
                        float(min_val),
                        float(max_val),
                        float((min_val + max_val) / 2)
                    )
            
            if st.button("🔮 Make Prediction", use_container_width=True):
                input_array = np.array([list(input_dict.values())])
                
                models = {
                    'Linear Regression': LinearRegression(),
                    'Decision Tree': DecisionTreeRegressor(random_state=42),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'SVR': SVR(kernel='rbf'),
                    'KNN': KNeighborsRegressor(n_neighbors=5)
                }
                
                predictions = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(input_array)[0]
                    predictions[name] = pred
                
                st.subheader("Predictions")
                cols = st.columns(len(predictions))
                
                for idx, (name, pred) in enumerate(predictions.items()):
                    with cols[idx]:
                        st.metric(name, f"{pred:.2f}")
                
                avg_pred = np.mean(list(predictions.values()))
                st.info(f"**Average Prediction:** {avg_pred:.2f}")

else:
    st.error("Unable to load the dataset. Please check the file path.")

st.markdown("---")
st.markdown("<center>Engine Failure ML Dashboard | Powered by Streamlit</center>", unsafe_allow_html=True)
