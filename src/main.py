# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import os  # For interacting with the operating system

# Import custom modules for data preprocessing, model training, forecasting, and visualization
from data_preprocessing import load_data, preprocess_data
from forecasting_model import train_model, forecast
from visualization import plot_forecast


# Define a function to forecast and save the results
def forecast_and_save(
    filepath,
    columns_to_forecast,
    output_dir,
    output_filename,
    order,
    seasonal_order,
    periods=10,
):
    # Load and preprocess the data
    raw_data = load_data(filepath)
    processed_data = preprocess_data(raw_data)

    # Train the model
    model = train_model(processed_data, "Total", order, seasonal_order)

    # Forecast total number of students
    total_forecast = forecast(model, periods).round().astype(int)

    # Calculate proportions
    proportions = (
        processed_data[columns_to_forecast].div(processed_data["Total"], axis=0).mean()
    )

    # Generate future periods
    last_period = processed_data["Period"].iloc[-1]
    last_year, last_semester = map(int, last_period.split("-"))
    future_periods = []
    for _ in range(periods):
        if last_semester == 1:
            last_semester = 2
        else:
            last_semester = 1
            last_year += 1
        future_periods.append(f"{last_year}-{last_semester}")

    # Create a dictionary to store forecasts
    all_predictions = {"Period": future_periods, "Total": total_forecast}

    for column in columns_to_forecast:
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Define output directory and file path
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    # Save to CSV
    predictions_df.to_csv(output_filepath, index=False)
    print(f"All forecasts saved to {output_filepath}")

    return processed_data, all_predictions, future_periods


def forecast_sexo(filepath, total_forecast, output_dir, output_filename, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Calculate proportions
    proportions = (
        processed_data[["Hombres", "Mujeres"]]
        .div(processed_data["Total"], axis=0)
        .mean()
    )  # Calculate the mean proportions for 'Hombres' and 'Mujeres'

    # Generate future periods
    last_period = processed_data["Period"].iloc[
        -1
    ]  # Get the last period from the processed data
    last_year, last_semester = map(
        int, last_period.split("-")
    )  # Split the last period into year and semester
    future_periods = []  # Initialize an empty list to store future periods
    for _ in range(periods):  # Loop through the number of periods to forecast
        if last_semester == 1:  # If the last semester is 1
            last_semester = 2  # Set the next semester to 2
        else:  # If the last semester is not 1
            last_semester = 1  # Set the next semester to 1
            last_year += 1  # Increment the year by 1
        future_periods.append(
            f"{last_year}-{last_semester}"
        )  # Append the new period to the future periods list

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast,
    }  # Initialize the dictionary with future periods and total forecast

    for column in ["Hombres", "Mujeres"]:  # Loop through each column to forecast
        # Forecast using proportions
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )  # Calculate the forecast for each column using the proportions

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it does not exist
    output_filepath = os.path.join(
        output_dir, output_filename
    )  # Define the full output file path

    # Save to CSV
    predictions_df.to_csv(
        output_filepath, index=False
    )  # Save the predictions DataFrame to a CSV file
    print(
        f"All forecasts saved to {output_filepath}"
    )  # Print a message indicating that the forecasts have been saved

    return (
        processed_data,
        all_predictions,
        future_periods,
    )  # Return the processed data, all predictions, and future periods


def forecast_sede(filepath, total_forecast, output_dir, output_filename, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Calculate proportions
    sede_columns = [
        "Amazonía",
        "Bogotá",
        "Caribe",
        "La Paz",
        "Manizales",
        "Medellín",
        "Orinoquía",
        "Palmira",
        "Tumaco",
    ]  # List of columns representing different campuses
    proportions = (
        processed_data[sede_columns].div(processed_data["Total"], axis=0).mean()
    )  # Calculate the mean proportions for each campus

    # Generate future periods
    last_period = processed_data["Period"].iloc[
        -1
    ]  # Get the last period from the processed data
    last_year, last_semester = map(
        int, last_period.split("-")
    )  # Split the last period into year and semester
    future_periods = []  # Initialize an empty list to store future periods
    for _ in range(periods):  # Loop through the number of periods to forecast
        if last_semester == 1:  # If the last semester is 1
            last_semester = 2  # Set the next semester to 2
        else:  # If the last semester is not 1
            last_semester = 1  # Set the next semester to 1
            last_year += 1  # Increment the year by 1
        future_periods.append(
            f"{last_year}-{last_semester}"
        )  # Append the new period to the future periods list

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast,
    }  # Initialize the dictionary with future periods and total forecast

    for column in sede_columns:  # Loop through each campus column to forecast
        # Forecast using proportions
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )  # Calculate the forecast for each campus using the proportions

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it does not exist
    output_filepath = os.path.join(
        output_dir, output_filename
    )  # Define the full output file path

    # Save to CSV
    predictions_df.to_csv(
        output_filepath, index=False
    )  # Save the predictions DataFrame to a CSV file
    print(
        f"All forecasts saved to {output_filepath}"
    )  # Print a message indicating that the forecasts have been saved

    return (
        processed_data,
        all_predictions,
        future_periods,
    )  # Return the processed data, all predictions, and future periods


def forecast_area(filepath, total_forecast, output_dir, output_filename, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)
    processed_data = preprocess_data(raw_data)

    # Calculate proportions
    area_columns = [
        "Administración de empresas y derecho",
        "Agricultura, silvicultura, pesca y veterinaria",
        "Artes y humanidades",
        "Ciencias naturales, matemáticas y estadística",
        "Ciencias sociales, periodismo e información",
        "Educación",
        "Ingeniería, industria y construcción",
        "Salud y bienestar",
        "Sin información",
        "Tecnologías de la información y la comunicación (TIC)",
    ]
    proportions = (
        processed_data[area_columns].div(processed_data["Total"], axis=0).mean()
    )

    # Generate future periods
    last_period = processed_data["Period"].iloc[-1]
    last_year, last_semester = map(int, last_period.split("-"))
    future_periods = []
    for _ in range(periods):
        if last_semester == 1:
            last_semester = 2
        else:
            last_semester = 1
            last_year += 1
        future_periods.append(f"{last_year}-{last_semester}")

    # Extract forecasted values from the SARIMAXResultsWrapper object
    total_forecast_values = (
        total_forecast.get_forecast(steps=periods).predicted_mean.round().astype(int)
    )
    total_forecast_values = total_forecast_values.reset_index(
        drop=True
    )  # Reset the index

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast_values,
    }
    for column in area_columns:
        all_predictions[column] = []

    # Forecast for each area using proportions
    for i in range(periods):
        for column in area_columns:
            area_forecast = (
                (total_forecast_values[i] * proportions[column]).round().astype(int)
            )
            all_predictions[column].append(area_forecast)

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Define output directory and file path
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    # Save to CSV
    predictions_df.to_csv(output_filepath, index=False)
    print(f"All forecasts saved to {output_filepath}")

    return processed_data, all_predictions, future_periods


def forecast_estadisticas(filepath, model, output_dir, output_filename, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)
    processed_data = preprocess_data(raw_data)

    # Calculate proportions
    proportions = (
        processed_data[["Postgrado", "Pregrado"]]
        .div(processed_data["Total"], axis=0)
        .mean()
    )

    # Generate future periods
    last_period = processed_data["Period"].iloc[-1]
    last_year, last_semester = map(int, last_period.split("-"))
    future_periods = []
    for _ in range(periods):
        if last_semester == 1:
            last_semester = 2
        else:
            last_semester = 1
            last_year += 1
        future_periods.append(f"{last_year}-{last_semester}")

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": [],
    }
    for column in ["Postgrado", "Pregrado"]:
        all_predictions[column] = []

    # Forecast one period at a time
    for period in range(periods):
        next_forecast = round(model.get_forecast(steps=1).predicted_mean.iloc[0])
        all_predictions["Total"].append(next_forecast)

        for column in ["Postgrado", "Pregrado"]:
            category_forecast = (
                (next_forecast * proportions[column]).round().astype(int)
            )
            all_predictions[column].append(category_forecast)

        model = model.append([next_forecast])

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Define output directory and file path
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    # Save to CSV
    predictions_df.to_csv(output_filepath, index=False)
    print(f"All forecasts saved to {output_filepath}")

    return processed_data, all_predictions, future_periods


def forecast_estadisticas_admitidos(
    filepath, model, output_dir, output_filename, periods=10
):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Calculate proportions
    proportions = (
        processed_data[["Postgrado", "Pregrado"]]
        .div(processed_data["Total"], axis=0)
        .mean()
    )  # Calculate the mean proportions for 'Postgrado' and 'Pregrado'

    # Generate future periods
    last_period = processed_data["Period"].iloc[
        -1
    ]  # Get the last period from the processed data
    last_year, last_semester = map(
        int, last_period.split("-")
    )  # Split the last period into year and semester
    future_periods = []  # Initialize an empty list to store future periods
    for _ in range(periods):  # Loop through the number of periods to forecast
        if last_semester == 1:  # If the last semester is 1
            last_semester = 2  # Set the next semester to 2
        else:  # If the last semester is not 1
            last_semester = 1  # Set the next semester to 1
            last_year += 1  # Increment the year by 1
        future_periods.append(
            f"{last_year}-{last_semester}"
        )  # Append the new period to the future periods list

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": [],
        "Total": [],
    }  # Initialize the dictionary with future periods and total forecast
    for column in ["Postgrado", "Pregrado"]:
        all_predictions[column] = []

    # Forecast one period at a time
    for period in range(periods):
        # Forecast total number of students for the next period
        next_forecast = model.get_forecast(steps=1).predicted_mean.iloc[
            0
        ]  # Forecast the next period
        all_predictions["Period"].append(future_periods[period])
        all_predictions["Total"].append(next_forecast)

        # Forecast for each category using proportions
        for column in ["Postgrado", "Pregrado"]:
            category_forecast = (
                (next_forecast * proportions[column]).round().astype(int)
            )
            all_predictions[column].append(category_forecast)

        # Update the model with the new forecasted value
        model = model.append([next_forecast])

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it does not exist
    output_filepath = os.path.join(
        output_dir, output_filename
    )  # Define the full output file path

    # Save to CSV
    predictions_df.to_csv(
        output_filepath, index=False
    )  # Save the predictions DataFrame to a CSV file
    print(
        f"All forecasts saved to {output_filepath}"
    )  # Print a message indicating that the forecasts have been saved

    return (
        processed_data,
        all_predictions,
        future_periods,
    )  # Return the processed data, all predictions, and future periods


def forecast_estadisticas_aspirantes(
    filepath, model, output_dir, output_filename, periods=10
):
    # Load and preprocess the data
    raw_data = load_data(filepath)
    processed_data = preprocess_data(raw_data)

    # Calculate proportions
    proportions = (
        processed_data[["Postgrado", "Pregrado"]]
        .div(processed_data["Total"], axis=0)
        .mean()
    )

    # Generate future periods
    last_period = processed_data["Period"].iloc[-1]
    last_year, last_semester = map(int, last_period.split("-"))
    future_periods = []
    for _ in range(periods):
        if last_semester == 1:
            last_semester = 2
        else:
            last_semester = 1
            last_year += 1
        future_periods.append(f"{last_year}-{last_semester}")

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": [],
        "Total": [],
    }
    for column in ["Postgrado", "Pregrado"]:
        all_predictions[column] = []

    # Forecast one period at a time
    for period in range(periods):
        next_forecast = model.get_forecast(steps=1).predicted_mean.iloc[0]
        all_predictions["Period"].append(future_periods[period])
        all_predictions["Total"].append(next_forecast)

        for column in ["Postgrado", "Pregrado"]:
            category_forecast = (
                (next_forecast * proportions[column]).round().astype(int)
            )
            all_predictions[column].append(category_forecast)

        model = model.append([next_forecast])

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Define output directory and file path
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    # Save to CSV
    predictions_df.to_csv(output_filepath, index=False)
    print(f"All forecasts saved to {output_filepath}")

    return processed_data, all_predictions, future_periods


def forecast_lugar_procedencia(
    filepath, total_forecast, output_dir, output_filename, periods=10
):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Calculate proportions for each department
    department_proportions = (
        processed_data.groupby("Departamento")["Total"]
        .sum()
        .div(processed_data["Total"].sum())
    )  # Calculate the proportions for each department based on the total

    # Generate future periods
    last_period = processed_data["Period"].iloc[
        -1
    ]  # Get the last period from the processed data
    last_year, last_semester = map(
        int, last_period.split("-")
    )  # Split the last period into year and semester
    future_periods = []  # Initialize an empty list to store future periods
    for _ in range(periods):  # Loop through the number of periods to forecast
        if last_semester == 1:  # If the last semester is 1
            last_semester = 2  # Set the next semester to 2
        else:  # If the last semester is not 1
            last_semester = 1  # Set the next semester to 1
            last_year += 1  # Increment the year by 1
        future_periods.append(
            f"{last_year}-{last_semester}"
        )  # Append the new period to the future periods list

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast,
    }  # Initialize the dictionary with future periods and total forecast

    # Forecast for each department
    for department in department_proportions.index:  # Loop through each department
        department_forecast = (
            total_forecast * department_proportions[department]
        ).astype(
            int
        )  # Calculate the forecast for each department using the proportions
        all_predictions[department] = (
            department_forecast  # Add the forecast to the dictionary
        )

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it does not exist
    output_filepath = os.path.join(
        output_dir, output_filename
    )  # Define the full output file path

    # Save to CSV
    predictions_df.to_csv(
        output_filepath, index=False
    )  # Save the predictions DataFrame to a CSV file
    print(
        f"All forecasts saved to {output_filepath}"
    )  # Print a message indicating that the forecasts have been saved

    return (
        processed_data,
        all_predictions,
        future_periods,
    )  # Return the processed data, all predictions, and future periods


def forecast_lugar_nacimiento(
    filepath, total_forecast, output_dir, output_filename, periods=10
):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Calculate proportions for each department
    department_proportions = (
        processed_data.groupby("Departamento")["Total"]
        .sum()
        .div(processed_data["Total"].sum())
    )  # Calculate the proportions for each department based on the total

    # Generate future periods
    last_period = processed_data["Period"].iloc[
        -1
    ]  # Get the last period from the processed data
    last_year, last_semester = map(
        int, last_period.split("-")
    )  # Split the last period into year and semester
    future_periods = []  # Initialize an empty list to store future periods
    for _ in range(periods):  # Loop through the number of periods to forecast
        if last_semester == 1:  # If the last semester is 1
            last_semester = 2  # Set the next semester to 2
        else:  # If the last semester is not 1
            last_semester = 1  # Set the next semester to 1
            last_year += 1  # Increment the year by 1
        future_periods.append(
            f"{last_year}-{last_semester}"
        )  # Append the new period to the future periods list

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast,
    }  # Initialize the dictionary with future periods and total forecast

    # Forecast for each department
    for department in department_proportions.index:  # Loop through each department
        department_forecast = (
            total_forecast * department_proportions[department]
        ).astype(
            int
        )  # Calculate the forecast for each department using the proportions
        all_predictions[department] = (
            department_forecast  # Add the forecast to the dictionary
        )

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it does not exist
    output_filepath = os.path.join(
        output_dir, output_filename
    )  # Define the full output file path

    # Save to CSV
    predictions_df.to_csv(
        output_filepath, index=False
    )  # Save the predictions DataFrame to a CSV file
    print(
        f"All forecasts saved to {output_filepath}"
    )  # Print a message indicating that the forecasts have been saved

    return (
        processed_data,
        all_predictions,
        future_periods,
    )  # Return the processed data, all predictions, and future periods


def forecast_modalidad(
    filepath, total_forecast, output_dir, output_filename, periods=10
):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Calculate proportions for each modality
    modality_columns = [
        "Doctorado",
        "Especialidades médicas",
        "Especialización",
        "Maestría",
        "Pregrado",
    ]  # List of columns representing different modalities
    proportions = (
        processed_data[modality_columns].div(processed_data["Total"], axis=0).mean()
    )  # Calculate the mean proportions for each modality

    # Generate future periods
    last_period = processed_data["Period"].iloc[
        -1
    ]  # Get the last period from the processed data
    last_year, last_semester = map(
        int, last_period.split("-")
    )  # Split the last period into year and semester
    future_periods = []  # Initialize an empty list to store future periods
    for _ in range(periods):  # Loop through the number of periods to forecast
        if last_semester == 1:  # If the last semester is 1
            last_semester = 2  # Set the next semester to 2
        else:  # If the last semester is not 1
            last_semester = 1  # Set the next semester to 1
            last_year += 1  # Increment the year by 1
        future_periods.append(
            f"{last_year}-{last_semester}"
        )  # Append the new period to the future periods list

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast,
    }  # Initialize the dictionary with future periods and total forecast

    for column in modality_columns:  # Loop through each modality column to forecast
        # Forecast using proportions
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )  # Calculate the forecast for each modality using the proportions

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it does not exist
    output_filepath = os.path.join(
        output_dir, output_filename
    )  # Define the full output file path

    # Save to CSV
    predictions_df.to_csv(
        output_filepath, index=False
    )  # Save the predictions DataFrame to a CSV file
    print(
        f"All forecasts saved to {output_filepath}"
    )  # Print a message indicating that the forecasts have been saved

    return (
        processed_data,
        predictions_df,
        future_periods,
    )  # Return the processed data, predictions DataFrame, and future periods


def forecast_nacionalidad(
    filepath, total_forecast, output_dir, output_filename, periods=10
):
    # Load and preprocess the data
    raw_data = load_data(filepath)
    processed_data = preprocess_data(raw_data)

    # Calculate proportions for each nationality
    nationality_columns = [
        "Colombiana",
        "Extranjero",
        "Sin información",
    ]
    proportions = (
        processed_data[nationality_columns].div(processed_data["Total"], axis=0).mean()
    )

    # Generate future periods
    last_period = processed_data["Period"].iloc[-1]
    last_year, last_semester = map(int, last_period.split("-"))
    future_periods = []
    for _ in range(periods):
        if last_semester == 1:
            last_semester = 2
        else:
            last_semester = 1
            last_year += 1
        future_periods.append(f"{last_year}-{last_semester}")

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast.round().astype(int),
    }
    for column in nationality_columns:
        all_predictions[column] = []

    # Forecast using proportions
    for column in nationality_columns:
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Define output directory and file path
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    # Save to CSV
    predictions_df.to_csv(output_filepath, index=False)
    print(f"All forecasts saved to {output_filepath}")

    return processed_data, all_predictions, future_periods


def forecast_estrato(filepath, total_forecast, output_dir, output_filename, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)
    processed_data = preprocess_data(raw_data)

    # Calculate proportions
    estrato_columns = [
        "Estrato 1",
        "Estrato 2",
        "Estrato 3",
        "Estrato 4",
        "Estrato 5",
        "Estrato 6",
        "ND/NE",
    ]
    # Ensure all columns exist in the dataset
    existing_columns = [col for col in estrato_columns if col in processed_data.columns]
    proportions = (
        processed_data[existing_columns].div(processed_data["Total"], axis=0).mean()
    )

    # Generate future periods
    last_period = processed_data["Period"].iloc[-1]
    last_year, last_semester = map(int, last_period.split("-"))
    future_periods = []
    for _ in range(periods):
        if last_semester == 1:
            last_semester = 2
        else:
            last_semester = 1
            last_year += 1
        future_periods.append(f"{last_year}-{last_semester}")

    # Create a dictionary to store forecasts
    all_predictions = {
        "Period": future_periods,
        "Total": total_forecast.round().astype(int),
    }
    for column in existing_columns:
        all_predictions[column] = []

    # Forecast for each category using proportions
    for column in existing_columns:
        category_forecast = (total_forecast * proportions[column]).round().astype(int)
        all_predictions[column] = category_forecast

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Define output directory and file path
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    # Save to CSV
    predictions_df.to_csv(output_filepath, index=False)
    print(f"All forecasts saved to {output_filepath}")

    return processed_data, all_predictions, future_periods


def main():
    # Base output directory
    base_output_dir = "data/predictions"

    # File paths and columns to forecast
    matriculados_filepath = "data/csv/Matriculados Primera Vez.csv"
    sexo_filepath = "data/csv/Sexo.csv"
    sede_filepath = "data/csv/Sede.csv"
    area_filepath = "data/csv/Area del Conocimiento.csv"
    estadisticas_filepath = "data/csv/Estadisticas Matriculados.csv"
    estadisticas_admitidos_filepath = "data/csv/Estadisticas Admitidos.csv"
    lugar_nacimiento_filepath = "data/csv/Lugar Nacimiento.csv"
    lugar_procedencia_filepath = "data/csv/Lugar Procedencia.csv"
    modalidad_filepath = "data/csv/Modalidad.csv"
    nacionalidad_filepath = "data/csv/Nacionalidad.csv"
    estrato_filepath = "data/csv/Estrato.csv"
    matriculados_columns = [
        "No",
        "Sí",
    ]

    # Forecast and save for the "Matriculados Primera Vez" dataset
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 2)
    periods = 10
    matriculados_output_dir = os.path.join(base_output_dir, "Matriculados_Primera_Vez")
    matriculados_data, matriculados_predictions, matriculados_future_periods = (
        forecast_and_save(
            matriculados_filepath,
            matriculados_columns,
            matriculados_output_dir,
            "Matriculados_Primera_Vez_forecasts.csv",
            order,
            seasonal_order,
            periods,
        )
    )

    # Train the model for the "Area" dataset
    area_raw_data = load_data(area_filepath)
    area_processed_data = preprocess_data(area_raw_data)
    area_model = train_model(area_processed_data, "Total", order, seasonal_order)

    # Forecast and save for the "Area" dataset using the trained model
    area_output_dir = os.path.join(base_output_dir, "Area")
    area_data, area_predictions, area_future_periods = forecast_area(
        area_filepath, area_model, area_output_dir, "Area_forecasts.csv", periods
    )

    # Forecast and save for the "Sexo" dataset using the total forecast from "Matriculados Primera Vez"
    total_forecast = matriculados_predictions["Total"]
    sexo_output_dir = os.path.join(base_output_dir, "Sexo")
    sexo_data, sexo_predictions, sexo_future_periods = forecast_sexo(
        sexo_filepath, total_forecast, sexo_output_dir, "Sexo_forecasts.csv", periods
    )

    # Forecast and save for the "Sede" dataset using the total forecast from "Matriculados Primera Vez"
    sede_output_dir = os.path.join(base_output_dir, "Sede")
    sede_data, sede_predictions, sede_future_periods = forecast_sede(
        sede_filepath, total_forecast, sede_output_dir, "Sede_forecasts.csv", periods
    )

    # Forecast and save for the "Estadisticas Matriculados" dataset
    estadisticas_output_dir = os.path.join(base_output_dir, "Estadisticas_Matriculados")
    estadisticas_model = train_model(
        area_processed_data, "Total", order, seasonal_order
    )
    estadisticas_data, estadisticas_predictions, estadisticas_future_periods = (
        forecast_estadisticas(
            estadisticas_filepath,
            estadisticas_model,
            estadisticas_output_dir,
            "Estadisticas_forecasts.csv",
            periods,
        )
    )

    # Forecast and save for the "Estadisticas Admitidos" dataset
    estadisticas_admitidos_output_dir = os.path.join(
        base_output_dir, "Estadisticas_Admitidos"
    )
    estadisticas_admitidos_model = train_model(
        load_data(estadisticas_admitidos_filepath), "Total", order, seasonal_order
    )
    (
        estadisticas_admitidos_data,
        estadisticas_admitidos_predictions,
        estadisticas_admitidos_future_periods,
    ) = forecast_estadisticas_admitidos(
        estadisticas_admitidos_filepath,
        estadisticas_admitidos_model,
        estadisticas_admitidos_output_dir,
        "Estadisticas_Admitidos_forecasts.csv",
        periods,
    )

    # File path for "Estadisticas Aspirantes" dataset
    estadisticas_aspirantes_filepath = "data/csv/Estadisticas Aspirantes.csv"

    # Forecast and save for the "Estadisticas Aspirantes" dataset
    estadisticas_aspirantes_output_dir = os.path.join(
        base_output_dir, "Estadisticas_Aspirantes"
    )
    estadisticas_aspirantes_model = train_model(
        load_data(estadisticas_aspirantes_filepath), "Total", order, seasonal_order
    )
    (
        estadisticas_aspirantes_data,
        estadisticas_aspirantes_predictions,
        estadisticas_aspirantes_future_periods,
    ) = forecast_estadisticas_aspirantes(
        estadisticas_aspirantes_filepath,
        estadisticas_aspirantes_model,
        estadisticas_aspirantes_output_dir,
        "Estadisticas_Aspirantes_forecasts.csv",
        periods,
    )

    # Forecast and save for the "Lugar Nacimiento" dataset using the total forecast from "Matriculados Primera Vez"
    lugar_nacimiento_output_dir = os.path.join(base_output_dir, "Lugar_Nacimiento")
    (
        lugar_nacimiento_data,
        lugar_nacimiento_predictions,
        lugar_nacimiento_future_periods,
    ) = forecast_lugar_nacimiento(
        lugar_nacimiento_filepath,
        total_forecast,
        lugar_nacimiento_output_dir,
        "Lugar_Nacimiento_forecasts.csv",
        periods,
    )

    # Forecast and save for the "Lugar Procedencia" dataset using the total forecast from "Matriculados Primera Vez"
    lugar_procedencia_output_dir = os.path.join(base_output_dir, "Lugar_Procedencia")
    (
        lugar_procedencia_data,
        lugar_procedencia_predictions,
        lugar_procedencia_future_periods,
    ) = forecast_lugar_procedencia(
        lugar_procedencia_filepath,
        total_forecast,
        lugar_procedencia_output_dir,
        "Lugar_Procedencia_forecasts.csv",
        periods,
    )

    # Forecast and save for the "Modalidad" dataset using the total forecast from "Matriculados Primera Vez"
    modalidad_output_dir = os.path.join(base_output_dir, "Modalidad")
    modalidad_data, modalidad_predictions, modalidad_future_periods = (
        forecast_modalidad(
            modalidad_filepath,
            total_forecast,
            modalidad_output_dir,
            "Modalidad_forecasts.csv",
            periods,
        )
    )

    # Forecast and save for the "Nacionalidad" dataset using the total forecast from "Matriculados Primera Vez"
    nacionalidad_output_dir = os.path.join(base_output_dir, "Nacionalidad")
    nacionalidad_data, nacionalidad_predictions, nacionalidad_future_periods = (
        forecast_nacionalidad(
            nacionalidad_filepath,
            total_forecast,
            nacionalidad_output_dir,
            "Nacionalidad_forecasts.csv",
            periods,
        )
    )

    # Forecast and save for the "Estrato" dataset using the total forecast from "Matriculados Primera Vez"
    estrato_output_dir = os.path.join(base_output_dir, "Estrato")
    (
        estrato_data,
        estrato_predictions,
        estrato_future_periods,
    ) = forecast_estrato(
        estrato_filepath,
        total_forecast,
        estrato_output_dir,
        "Estrato_forecasts.csv",
        periods,
    )

    # Define the output directory for plots
    output_dir = "data/predictions"

    # Menu options
    menu_options = [
        ("Area", area_data, area_predictions, area_future_periods, area_output_dir),
        (
            "Estadisticas Matriculados",
            estadisticas_data,
            estadisticas_predictions,
            estadisticas_future_periods,
            estadisticas_output_dir,
        ),
        (
            "Estadisticas Admitidos",
            estadisticas_admitidos_data,
            estadisticas_admitidos_predictions,
            estadisticas_admitidos_future_periods,
            estadisticas_admitidos_output_dir,
        ),
        (
            "Estadisticas Aspirantes",
            estadisticas_aspirantes_data,
            estadisticas_aspirantes_predictions,
            estadisticas_aspirantes_future_periods,
            estadisticas_aspirantes_output_dir,
        ),
        (
            "Lugar Nacimiento",
            lugar_nacimiento_data,
            lugar_nacimiento_predictions,
            lugar_nacimiento_future_periods,
            lugar_nacimiento_output_dir,
        ),
        (
            "Lugar Procedencia",
            lugar_procedencia_data,
            lugar_procedencia_predictions,
            lugar_procedencia_future_periods,
            lugar_procedencia_output_dir,
        ),
        (
            "Matriculados Primera Vez",
            matriculados_data,
            matriculados_predictions,
            matriculados_future_periods,
            matriculados_output_dir,
        ),
        (
            "Modalidad",
            modalidad_data,
            modalidad_predictions,
            modalidad_future_periods,
            modalidad_output_dir,
        ),
        (
            "Nacionalidad",
            nacionalidad_data,
            nacionalidad_predictions,
            nacionalidad_future_periods,
            nacionalidad_output_dir,
        ),
        ("Sede", sede_data, sede_predictions, sede_future_periods, sede_output_dir),
        ("Sexo", sexo_data, sexo_predictions, sexo_future_periods, sexo_output_dir),
        (
            "Estrato",
            estrato_data,
            estrato_predictions,
            estrato_future_periods,
            estrato_output_dir,
        ),
    ]

    # Sort menu options alphabetically by name
    menu_options.sort(key=lambda x: x[0])

    # Menu for user to choose which plot to see
    while True:
        os.system("cls")
        print("\nSelect the dataset to plot the forecast:")
        for i, (name, _, _, _, _) in enumerate(menu_options, 1):
            print(f"{i}. {name}")
        print("0. Exit")

        dataset_choice = int(input("Enter the number of your choice: "))

        if dataset_choice == 0:
            print("Exiting the program.")
            break
        elif 1 <= dataset_choice <= len(menu_options):
            selected_option = menu_options[dataset_choice - 1]
            name, processed_data, all_predictions, future_periods, output_dir = (
                selected_option
            )

            if name == "Area":
                columns_to_forecast = [
                    "Administración de empresas y derecho",
                    "Agricultura, silvicultura, pesca y veterinaria",
                    "Artes y humanidades",
                    "Ciencias naturales, matemáticas y estadística",
                    "Ciencias sociales, periodismo e información",
                    "Educación",
                    "Ingeniería, industria y construcción",
                    "Salud y bienestar",
                    "Sin información",
                    "Tecnologías de la información y la comunicación (TIC)",
                ]
            elif (
                name == "Estadisticas Matriculados"
                or name == "Estadisticas Admitidos"
                or name == "Estadisticas Aspirantes"
            ):
                columns_to_forecast = ["Postgrado", "Pregrado"]
            elif name == "Lugar Nacimiento" or name == "Lugar Procedencia":
                columns_to_forecast = processed_data["Departamento"].unique()
            elif name == "Matriculados Primera Vez":
                columns_to_forecast = matriculados_columns
            elif name == "Modalidad":
                columns_to_forecast = [
                    "Doctorado",
                    "Especialidades médicas",
                    "Especialización",
                    "Maestría",
                    "Pregrado",
                ]
            elif name == "Nacionalidad":
                columns_to_forecast = ["Colombiana", "Extranjero", "Sin información"]
            elif name == "Sede":
                columns_to_forecast = [
                    "Amazonía",
                    "Bogotá",
                    "Caribe",
                    "La Paz",
                    "Manizales",
                    "Medellín",
                    "Orinoquía",
                    "Palmira",
                    "Tumaco",
                ]
            elif name == "Sexo":
                columns_to_forecast = ["Hombres", "Mujeres"]
            elif name == "Estrato":
                columns_to_forecast = [
                    "Estrato 1",
                    "Estrato 2",
                    "Estrato 3",
                    "Estrato 4",
                    "Estrato 5",
                    "Estrato 6",
                    "ND/NE",
                ]

            while True:
                print("\nSelect the column to plot the forecast:")
                for i, column in enumerate(columns_to_forecast, 1):
                    print(f"{i}. {column}")
                print(f"{len(columns_to_forecast) + 1}. Total")
                print("0. Back to dataset selection")

                column_choice = int(input("Enter the number of your choice: "))

                if column_choice == 0:
                    break
                elif 1 <= column_choice <= len(columns_to_forecast):
                    selected_column = columns_to_forecast[column_choice - 1]
                    try:
                        print(f"Plotting forecast for column: {selected_column}")
                        print(f"Processed data columns: {processed_data.columns}")
                        print(f"All predictions keys: {all_predictions.keys()}")
                        plot_forecast(
                            processed_data,
                            all_predictions[selected_column],
                            future_periods,
                            selected_column.replace("/", "_"),  # Replace "/" with "_" in column name
                            output_dir,
                        )
                    except KeyError:
                        print(f"An error occurred while plotting the forecast: '{selected_column}' not found.")
                elif column_choice == len(columns_to_forecast) + 1:
                    plot_forecast(
                        processed_data,
                        all_predictions["Total"],
                        future_periods,
                        "Total",
                        output_dir,
                    )
                else:
                    print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()