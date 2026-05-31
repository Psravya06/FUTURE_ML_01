import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("sales_data.csv")

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'])

# Sort by date
data = data.sort_values('Date')

# Create time index
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Features and target
X = data[['Days']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Error
mae = mean_absolute_error(y, predictions)

print("Mean Absolute Error:", mae)

# Future forecasting
future_days = np.arange(
    data['Days'].max() + 1,
    data['Days'].max() + 31
).reshape(-1, 1)

future_sales = model.predict(future_days)

# Plot
plt.figure(figsize=(10, 5))

plt.plot(
    data['Date'],
    y,
    label="Actual Sales"
)

future_dates = pd.date_range(
    start=data['Date'].max() + pd.Timedelta(days=1),
    periods=30
)

plt.plot(
    future_dates,
    future_sales,
    label="Forecasted Sales"
)

plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()

plt.show()