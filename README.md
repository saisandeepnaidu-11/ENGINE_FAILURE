# 🚗 Engine Failure ML Dashboard

A Streamlit-based web application for analyzing engine failure data using machine learning models.

## Features

- Interactive data visualization and analysis
- Multiple regression models (Linear Regression, Decision Tree, Random Forest, SVR, KNN)
- Model performance comparison
- Real-time predictions

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

4. Install the required packages:
   ```bash
   pip install streamlit pandas numpy seaborn matplotlib scikit-learn
   ```

## Usage

1. Ensure you have the `engine_failure_dataset.csv` file in the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run "sample (1).py"
   ```
3. Open your web browser and go to `http://localhost:8501`
4. Upload the dataset through the app interface and explore the analysis

## Dataset

The application expects a CSV file with the following columns:
- RPM
- Torque
- Fuel_Efficiency
- Power_Output (kW)
- Vibration_X, Vibration_Y, Vibration_Z
- Temperature (°C) - target variable

## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

## Contributing

Feel free to submit issues and enhancement requests!
