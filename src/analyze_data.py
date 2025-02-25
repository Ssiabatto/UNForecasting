import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import statsmodels.api as sm  # Import statsmodels for statistical modeling
from statsmodels.tsa.stattools import (
    adfuller,
)  # Import adfuller for Augmented Dickey-Fuller test
from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf,
)  # Import plot_acf and plot_pacf for ACF and PACF plots

# Load Data
df = pd.read_csv("data/csv/Matriculados Primera Vez.csv")  # Load the dataset

# Sort the data by 'Año' and 'Semestre'
df = df.sort_values(by=["Año", "Semestre"])  # Sort the dataframe by year and semester

# Create a datetime index
df["date"] = pd.to_datetime(
    df["Año"].astype(str) + "-" + (df["Semestre"] * 6).astype(str), format="%Y-%m"
)  # Create a datetime index with 6-month intervals
df.set_index("date", inplace=True)  # Set the datetime index

# Select total number of students
y = df["Total"]  # Select the 'Total' column as the time series

# ADF Test
result = adfuller(y)  # Perform the Augmented Dickey-Fuller test
print(f"ADF Statistic: {result[0]}")  # Print the ADF statistic
print(f"P-value: {result[1]}")  # Print the p-value

# Plot
plt.figure(figsize=(10, 4))  # Create a figure with specified size
plt.plot(y, marker="o", linestyle="-")  # Plot the time series
plt.title("Total Students Over Time")  # Set the title of the plot
plt.show()  # Show the plot

# Determine Differencing Order (d)
y_diff = y.diff().dropna()  # Difference the time series and drop missing values
result = adfuller(y_diff)  # Perform the ADF test on the differenced series
print(
    f"ADF Statistic (After Differencing): {result[0]}"
)  # Print the ADF statistic after differencing
print(f"P-value: {result[1]}")  # Print the p-value after differencing

plt.figure(figsize=(10, 4))  # Create a figure with specified size
plt.plot(y_diff, marker="o", linestyle="-", color="red")  # Plot the differenced series
plt.title("Differenced Data")  # Set the title of the plot
plt.show()  # Show the plot

# Check Seasonal Differencing (D)
y_seasonal_diff = y.diff(
    periods=2
).dropna()  # Perform seasonal differencing with a period of 2
result = adfuller(
    y_seasonal_diff
)  # Perform the ADF test on the seasonally differenced series
print(
    f"ADF Statistic (Seasonally Differenced): {result[0]}"
)  # Print the ADF statistic for seasonally differenced series
print(f"P-value: {result[1]}")  # Print the p-value for seasonally differenced series

# Further differencing if needed
if result[1] > 0.05:  # Check if the p-value is greater than 0.05
    y_seasonal_diff = (
        y_seasonal_diff.diff().dropna()
    )  # Further difference the series if needed
    result = adfuller(y_seasonal_diff)  # Perform the ADF test again
    print(
        f"ADF Statistic (Further Differenced): {result[0]}"
    )  # Print the ADF statistic after further differencing
    print(f"P-value: {result[1]}")  # Print the p-value after further differencing

# Find p, q, P, Q using ACF and PACF Plots
fig, ax = plt.subplots(2, 1, figsize=(12, 6))  # Create subplots for ACF and PACF plots
plot_acf(y_seasonal_diff, lags=10, ax=ax[0])  # Plot the ACF
plot_pacf(y_seasonal_diff, lags=10, ax=ax[1])  # Plot the PACF
plt.show()  # Show the plots

""" ADF Statistic: -2.6267023613177134
P-value: 0.08760305266789092
ADF Statistic (After Differencing): -5.679688315374852
P-value: 8.542877284146076e-07
ADF Statistic (Seasonally Differenced): -5.692909588208957
P-value: 7.994795892482983e-07

Example Interpretation of ACF and PACF Plots:
p: Count of significant lags in the PACF plot before it cuts off.
q: Count of significant lags in the ACF plot before it cuts off.
P: Count of significant seasonal lags in the PACF plot before it cuts off.
Q: Count of significant seasonal lags in the ACF plot before it cuts off.
Assume the following interpretation based on the plots:

ACF plot: Significant lags at 1 and 2 (q=1) "If we use 2, the amount of lags doesnt allow it to work properly"
PACF plot: Significant lags at 1 and 2 (p=1) "If we use 2, the amount of lags doesnt allow it to work properly"
Seasonal ACF plot: Significant lag at 1 (Q=1)
Seasonal PACF plot: Significant lag at 1 (P=1) """
