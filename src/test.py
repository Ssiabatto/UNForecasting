import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the sexo.csv file
filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Sexo.csv"
sexo_df = pd.read_csv(filepath)

# Preprocess the data if necessary
# Assuming the CSV has columns 'Año', 'Semestre', 'Hombres', 'Mujeres'
sexo_df['Period'] = sexo_df['Año'].astype(str) + '-' + sexo_df['Semestre'].astype(str)
df = sexo_df.sort_values(by=['Año', 'Semestre'])
df.set_index('Period', inplace=True)

# Access a particular column, for example, 'Hombres'
hombres_column = df["Hombres"]
hombres1 = df["Hombres"][0]
print("Sexo:")
print(df)
print(hombres1)

for i in range len(df):
    
percentages = [df["Hombres"][0]]

# Define your percentage data
percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # You can edit this list

# Generate some sample percentage data
dates = pd.date_range(start="2020-01-01", periods=len(percentages), freq="D")
data = np.array(percentages)
df = pd.DataFrame(data, index=dates, columns=["value"])

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fit SARIMAX model
model = SARIMAX(train["value"], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)

# Make predictions
predictions = model_fit.predict(
    start=len(train), end=len(train) + len(test) - 1, dynamic=False
)

# Print predicted percentages
print("Predicted Percentages:")
print(predictions)

# Calculate error
error = mean_squared_error(test["value"], predictions)
print(f"Test MSE: {error}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train["value"], label="Train")
plt.plot(test.index, test["value"], label="Test")
plt.plot(test.index, predictions, label="Predictions", color="red")
plt.legend()
plt.show()

# Print the number of predictions made
print(f"Number of predictions made: {len(predictions)}")
