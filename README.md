# Stock Prediction Web App

### Description

The project focuses on predicting stock trends by utilizing machine learning models and technical analysis indicators. It's designed to assist users in making informed decisions regarding stock investments. The user interacts with the system through a Streamlit web interface, which provides a seamless and intuitive experience.

### Libraries Used

- **numpy**: Utilized for its powerful numerical computation capabilities, such as array manipulation and mathematical operations.
- **plotly.graph_objs**: Employed for creating visualizations, specifically for plotting stock price data and technical indicators.
- **pandas_datareader**: Used to fetch historical stock price data from various online sources, facilitating data acquisition for analysis.
- **streamlit**: Chosen for building the user interface due to its simplicity and ease of use in creating interactive web applications.
- **yfinance**: Integrated for fetching stock data directly from Yahoo Finance, ensuring reliable and up-to-date information for analysis.
- **pandas**: Essential for data manipulation and analysis tasks, such as handling time series data and performing statistical operations.
- **keras.models**: Used to load pre-trained machine learning models, enabling seamless integration of predictive analytics into the project.
- **sklearn.preprocessing.MinMaxScaler**: Employed for scaling data to a specified range, ensuring consistency and effectiveness in model training.
- **datetime**: Utilized for handling date and time-related operations, essential for processing time series data and conducting analysis over specific time periods.

### Working

1. **Data Acquisition**: Fetches historical stock price data based on user input, including the stock ticker and start date for analysis.
2. **Data Exploration**: Describes the statistical properties of the data and visualizes the closing prices over time to provide insights into stock trends.
3. **Technical Analysis**: Calculates and plots moving averages (MA) as technical indicators to aid in identifying trends and potential trading opportunities.
4. **Model Training**: Splits the data into training and testing sets, scales the training data, and loads a pre-trained machine learning model for prediction.
5. **Prediction**: Predicts future stock prices using the trained model, allowing users to forecast stock trends over a specified number of weeks.

### Features

- **User Input**: Enables users to input a stock ticker symbol and start date for analysis, providing flexibility and customization.
- **Visualization**: Offers visual representations of historical stock prices, technical indicators, and predicted future prices to aid in decision-making.
- **Prediction**: Utilizes machine learning models to predict future stock prices, empowering users with insights into potential market movements.
- **User Interface**: Provides a user-friendly web interface powered by Streamlit, allowing users to interact with the system seamlessly and intuitively.
