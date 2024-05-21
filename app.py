import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
import yfinance as yf
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Fetch the stock price data based on user input start date
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_date_input = st.text_input('Enter Start Date (YYYY-MM-DD) minimum time should be 2 years', '2010-01-01')
start_date = datetime.strptime(start_date_input, '%Y-%m-%d')
end_date = datetime.now()

df = yf.download(user_input, start=start_date, end=end_date)

# Describing Data
st.subheader('Data from {} to {}'.format(start_date.date(), end_date.date()))
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
# Assuming the column name is 'Close' (case-sensitive)
plt.plot(df.index, df['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
# Assuming the column name is 'Close' (case-sensitive)
plt.plot(df.index, df['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
# Assuming the column name is 'Close' (case-sensitive)
plt.plot(df.index, df['Close'], 'b')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(fig)

# Split the data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

# scaling down the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load my model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)

# Concatenate the two DataFrames vertically
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='original price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Date')
plt.xlabel('Price')
plt.legend()
st.pyplot(fig2)

# Function to predict stock prices for the specified number of weeks
def predict_future_prices(weeks):
    future_dates = [end_date + timedelta(weeks=i) for i in range(1, weeks+1)]
    future_data = pd.DataFrame(index=pd.date_range(start=end_date + timedelta(days=1), periods=7*weeks, freq='D'), columns=['Predicted Price'])
    last_100_days = data_testing.tail(100).values.reshape(-1, 1)
    for i in range(weeks):
        for j in range(7):
            x_input = last_100_days[-100:]
            x_input = x_input.reshape((1, 100, 1))
            y_pred = model.predict(x_input)[0][0]
            future_data.iloc[i*7 + j]['Predicted Price'] = y_pred
            last_100_days = np.append(last_100_days, [[y_pred]], axis=0)
    return future_data

# Get user input for number of weeks
num_weeks = st.number_input('Enter number of weeks for future prediction', min_value=1, step=1)

# Predict future prices and display as a table
if st.button('Predict Future Prices'):
    future_prices = predict_future_prices(num_weeks)
    st.subheader('Predicted Stock Prices for the Next {} Weeks'.format(num_weeks))
    st.table(future_prices)
