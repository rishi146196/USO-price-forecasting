# Time Series Forecasting for Financial Markets using LSTM

## Overview
This project leverages deep learning techniques, particularly Long Short-Term Memory (LSTM) neural networks, to predict the future closing prices of the **United States Oil Fund (USO)**. By analyzing historical financial market data, the project aims to provide actionable insights for investors and decision-makers.

---

## Dataset
### Source:
The dataset used is **FINAL_USO.csv**, containing historical market data for key financial indices and ETFs, including:
- **SP (S&P 500)**: Opening, High, Low, Close, and Volume.
- **GDX (Gold Miners ETF)**: Prices and Volumes.
- **USO (United States Oil Fund)**: Prices and Volumes (target variable for forecasting).

### Key Features:
- Dates spanning multiple years of market activity.
- Relevant columns for each index/ETF: Open, High, Low, Close, Adjusted Close, and Volume.

---

## Project Workflow
### 1. Data Preprocessing
- Converted dates into datetime format and sorted data.
- Scaled the target variable (`USO_Close`) using **MinMaxScaler**.
- Handled missing values and ensured the dataset's integrity.

### 2. Exploratory Data Analysis (EDA)
- Visualized historical trends in `USO_Close` prices.
- Identified patterns and seasonality in the data.

### 3. Feature Engineering
- Created sequences of length 50 to train the LSTM model.
- Included rolling statistics and lagged features for potential additional insights.

### 4. Modeling
- Designed and trained an **LSTM model** with:
  - Two LSTM layers with 50 units each.
  - Dropout layers to prevent overfitting.
  - Dense output layer for single-step prediction.
- Used 80% of the data for training and 20% for testing.

### 5. Evaluation
- Metrics:
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Percentage Error (MAPE)**
- Compared actual vs. predicted values visually to assess performance.

### 6. Forecasting & Backtesting
- Forecasted the next 30 days of `USO_Close` prices.
- Simulated a basic trading strategy based on predicted price movements, achieving a significant portfolio value increase.

---

## Results
- **RMSE:** Low error, demonstrating the model's accuracy.
- **Visualization:** Predictions closely aligned with actual values.
- **Trading Strategy:** Backtesting showed positive returns based on model-driven decisions.

---

## Tools & Technologies
- **Python** for data processing and model development.
- Libraries:
  - **Pandas, Numpy:** Data manipulation and feature engineering.
  - **Matplotlib, Seaborn:** Data visualization.
  - **TensorFlow, Keras:** Building and training the LSTM model.
  - **Scikit-learn:** Scaling and evaluation metrics.



### 5. View Results
- Predicted vs. actual prices plotted.
- Forecasted prices for the next 30 days visualized.

---

## Future Improvements
- Experiment with other architectures (e.g., GRU, Transformer).
- Include additional features like macroeconomic indicators.
- Hyperparameter tuning using **KerasTuner** or **GridSearchCV**.
- Deploy the model via a Flask API or Streamlit web app for real-time predictions.

---



## Acknowledgements
- Financial market data source.
- TensorFlow and Keras documentation for deep learning insights.

---

Feel free to contribute to this project by submitting issues or pull requests!

