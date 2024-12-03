import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta

# Load the model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path and try again.")

st.title("Apple Stock Price Prediction")

# Specify the file path of your CSV data directly here
file_path = 'C:/Users/vedan/P452/apple_stock_data.csv'  # Change this to your actual file path

# Read the CSV file
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Data file not found. Please check the file path and try again.")

# Check if 'ds' column exists before proceeding
if 'ds' in data.columns:
    # Convert 'ds' column to datetime
    data['Date'] = pd.to_datetime(data['ds'])
    
    # Extract features for prediction
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # Define the latest stock data for Open, High, and Low prices
    last_known_open = data['Open'].iloc[-1]
    last_known_high = data['High'].iloc[-1]
    last_known_low = data['Low'].iloc[-1]

    # Define expected feature order based on the model training
    expected_features = ['Open', 'High', 'Low', 'Year', 'Month', 'Day']

    # Date input for user to predict a specific date within the next 30 days
    today = datetime.today()
    date_selected = st.date_input(
        "Select a date within the next 30 days for price prediction:",
        min_value=today, max_value=today + timedelta(days=30)
    )

    if st.button("Predict for Selected Date"):
        if date_selected:
            # Create features for the selected date
            selected_data = pd.DataFrame({
                'Open': [last_known_open],
                'High': [last_known_high],
                'Low': [last_known_low],
                'Year': [date_selected.year],
                'Month': [date_selected.month],
                'Day': [date_selected.day]
            })

            # Predict the price for the selected date
            selected_prediction = model.predict(selected_data)[0]
            st.write(f"Predicted Closing Price for {date_selected.strftime('%Y-%m-%d')}: ${selected_prediction:.2f}")

            # Generate predictions for the next 30 days starting from today
            future_dates = pd.date_range(start=today, periods=30, freq='D')
            future_data = pd.DataFrame({
                'Open': [last_known_open] * 30,
                'High': [last_known_high] * 30,
                'Low': [last_known_low] * 30,
                'Year': future_dates.year,
                'Month': future_dates.month,
                'Day': future_dates.day
            })

            # Predict for the next 30 days
            future_predictions = model.predict(future_data)

            # Display the next 30 days predictions
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Close': future_predictions
            })
            st.write("Predicted Closing Prices for the Next 30 Days:")
            st.write(forecast_df)

else:
    st.error("The uploaded file does not contain a 'ds' column.")
