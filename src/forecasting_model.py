import pandas as pd  # Import pandas for data manipulation
from statsmodels.tsa.statespace.sarimax import (
    SARIMAX,
)  # Import SARIMAX for time series forecasting


def train_model(df, column, order=(2, 1, 2), seasonal_order=(1, 1, 1, 2)):
    # Initialize the SARIMAX model with the specified order and seasonal order
    model = SARIMAX(
        df[column],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(
        disp=False
    )  # Fit the model to the data without displaying output
    return model_fit  # Return the fitted model


def forecast(model, periods):
    forecast = model.get_forecast(
        steps=periods
    )  # Generate forecast for the specified number of periods
    forecast_values = forecast.predicted_mean.round().astype(
        int
    )  # Round the forecasted values and convert to integers
    forecast_values[forecast_values < 0] = 0  # Replace negative values with zero
    return forecast_values  # Return the forecasted values