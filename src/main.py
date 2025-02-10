# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import os  # For interacting with the operating system

# Import custom modules for data preprocessing, model training, forecasting, and visualization
from data_preprocessing import load_data, preprocess_data
from forecasting_model import train_model, forecast
from visualization import plot_forecast


# Define a function to forecast and save the results
def forecast_and_save(
    filepath, columns_to_forecast, output_filename, order, seasonal_order, periods=10
):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Train the model
    model = train_model(processed_data, "Total", order, seasonal_order)
    # Train the forecasting model on the 'Total' column

    # Forecast total number of students
    total_forecast = forecast(model, periods)  # Forecast the total number of students
    # for the specified number of periods

    # Calculate proportions
    proportions = (
        processed_data[columns_to_forecast].div(processed_data["Total"], axis=0).mean()
    )
    # Calculate the mean proportions for each column to forecast

    # Generate future periods
    last_period = processed_data["Period"].iloc[-1]  # Get the last period from the
    # processed data
    last_year, last_semester = map(int, last_period.split("-"))  # Split the last period
    # into year and semester
    future_periods = []  # Initialize an empty list to store future periods
    for _ in range(periods):  # Loop through the number of periods to forecast
        if last_semester == 1:  # If the last semester is 1
            last_semester = 2  # Set the next semester to 2
        else:  # If the last semester is not 1
            last_semester = 1  # Set the next semester to 1
            last_year += 1  # Increment the year by 1
        future_periods.append(f"{last_year}-{last_semester}")  # Append the new
        # period to the future periods list

    # Create a dictionary to store forecasts
    all_predictions = {"Period": future_periods, "Total": total_forecast}  # Initialize
    # the dictionary with future periods and total forecast

    for column in columns_to_forecast:  # Loop through each column to forecast
        # Forecast using proportions
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )  # Calculate the forecast for each column using the proportions

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_sexo(filepath, total_forecast, output_filename, periods=10):
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
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_sede(filepath, total_forecast, output_filename, periods=10):
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
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_area(filepath, total_forecast, output_filename, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

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
    ]  # List of columns representing different areas of study
    proportions = (
        processed_data[area_columns].div(processed_data["Total"], axis=0).mean()
    )  # Calculate the mean proportions for each area of study

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

    for column in area_columns:  # Loop through each area column to forecast
        # Forecast using proportions
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )  # Calculate the forecast for each area using the proportions

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_estadisticas(filepath, output_filename, order, seasonal_order, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Train the model
    model = train_model(
        processed_data, "Total", order, seasonal_order
    )  # Train the forecasting model on the 'Total' column with specified order and seasonal order

    # Forecast total number of students
    total_forecast = forecast(
        model, periods
    )  # Forecast the total number of students for the specified number of periods

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
        "Period": future_periods,
        "Total": total_forecast,
    }  # Initialize the dictionary with future periods and total forecast

    for column in ["Postgrado", "Pregrado"]:  # Loop through each column to forecast
        # Forecast using proportions
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )  # Calculate the forecast for each column using the proportions

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_lugar_procedencia(filepath, total_forecast, output_filename, periods=10):
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
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_lugar_nacimiento(filepath, total_forecast, output_filename, periods=10):
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
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_modalidad(filepath, total_forecast, output_filename, periods=10):
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
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def forecast_nacionalidad(filepath, total_forecast, output_filename, periods=10):
    # Load and preprocess the data
    raw_data = load_data(filepath)  # Load the raw data from the specified file path
    processed_data = preprocess_data(raw_data)  # Preprocess the raw data

    # Calculate proportions for each nationality
    nationality_columns = [
        "Colombiana",
        "Extranjero",
        "Sin información",
    ]  # List of columns representing different nationalities
    proportions = (
        processed_data[nationality_columns].div(processed_data["Total"], axis=0).mean()
    )  # Calculate the mean proportions for each nationality

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

    for (
        column
    ) in nationality_columns:  # Loop through each nationality column to forecast
        # Forecast using proportions
        all_predictions[column] = (
            (total_forecast * proportions[column]).round().astype(int)
        )  # Calculate the forecast for each nationality using the proportions

    # Convert dictionary to DataFrame
    predictions_df = pd.DataFrame(
        all_predictions
    )  # Convert the dictionary of predictions to a DataFrame

    # Define output directory and file path
    output_dir = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/predictions"  # Define the output directory
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


def main():
    # File paths and columns to forecast
    matriculados_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Matriculados Primera Vez.csv"  # File path for "Matriculados Primera Vez" dataset
    sexo_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Sexo.csv"  # File path for "Sexo" dataset
    sede_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Sede.csv"  # File path for "Sede" dataset
    area_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Area.csv"  # File path for "Area" dataset
    estadisticas_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Estadisticas Matriculados UNAL.csv"  # File path for "Estadisticas Matriculados UNAL" dataset
    lugar_nacimiento_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Lugar Nacimiento.csv"  # File path for "Lugar Nacimiento" dataset
    lugar_procedencia_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Lugar Procedencia.csv"  # File path for "Lugar Procedencia" dataset
    modalidad_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Modalidad.csv"  # File path for "Modalidad" dataset
    nacionalidad_filepath = "d:/Archivos_Nicolás/UN/MateriasUN/2024-2/Modelos y Simulación/Proyecto_Modelos/Entrega 2/UNForecasting/data/csv/Nacionalidad.csv"  # File path for "Nacionalidad" dataset
    matriculados_columns = [
        "No",
        "Sí",
    ]  # Columns to forecast for "Matriculados Primera Vez" dataset

    # Forecast and save for the "Matriculados Primera Vez" dataset
    order = (1, 1, 1)  # Adjusted p, d, q values for the ARIMA model
    seasonal_order = (
        1,
        1,
        1,
        2,
    )  # Adjusted seasonal p, d, q, m values for the seasonal ARIMA model
    periods = 10  # Forecast 10 semesters ahead
    matriculados_data, matriculados_predictions, matriculados_future_periods = (
        forecast_and_save(
            matriculados_filepath,
            matriculados_columns,
            "Matriculados_forecasts.csv",
            order,
            seasonal_order,
            periods,
        )
    )

    # Forecast and save for the "Sexo" dataset using the total forecast from "Matriculados Primera Vez"
    total_forecast = matriculados_predictions[
        "Total"
    ]  # Use the total forecast from "Matriculados Primera Vez"
    sexo_data, sexo_predictions, sexo_future_periods = forecast_sexo(
        sexo_filepath, total_forecast, "Sexo_forecasts.csv", periods
    )

    # Forecast and save for the "Sede" dataset using the total forecast from "Matriculados Primera Vez"
    sede_data, sede_predictions, sede_future_periods = forecast_sede(
        sede_filepath, total_forecast, "Sede_forecasts.csv", periods
    )

    # Forecast and save for the "Area" dataset using the total forecast from "Matriculados Primera Vez"
    area_data, area_predictions, area_future_periods = forecast_area(
        area_filepath, total_forecast, "Area_forecasts.csv", periods
    )

    # Forecast and save for the "Estadisticas Matriculados UNAL" dataset
    estadisticas_data, estadisticas_predictions, estadisticas_future_periods = (
        forecast_estadisticas(
            estadisticas_filepath,
            "Estadisticas_forecasts.csv",
            order,
            seasonal_order,
            periods,
        )
    )

    # Forecast and save for the "Lugar Nacimiento" dataset using the total forecast from "Matriculados Primera Vez"
    (
        lugar_nacimiento_data,
        lugar_nacimiento_predictions,
        lugar_nacimiento_future_periods,
    ) = forecast_lugar_nacimiento(
        lugar_nacimiento_filepath,
        total_forecast,
        "Lugar_Nacimiento_forecasts.csv",
        periods,
    )

    # Forecast and save for the "Lugar Procedencia" dataset using the total forecast from "Matriculados Primera Vez"
    (
        lugar_procedencia_data,
        lugar_procedencia_predictions,
        lugar_procedencia_future_periods,
    ) = forecast_lugar_procedencia(
        lugar_procedencia_filepath,
        total_forecast,
        "Lugar_Procedencia_forecasts.csv",
        periods,
    )

    # Forecast and save for the "Modalidad" dataset using the total forecast from "Matriculados Primera Vez"
    modalidad_data, modalidad_predictions, modalidad_future_periods = (
        forecast_modalidad(
            modalidad_filepath, total_forecast, "Modalidad_forecasts.csv", periods
        )
    )

    # Forecast and save for the "Nacionalidad" dataset using the total forecast from "Matriculados Primera Vez"
    nacionalidad_data, nacionalidad_predictions, nacionalidad_future_periods = (
        forecast_nacionalidad(
            nacionalidad_filepath, total_forecast, "Nacionalidad_forecasts.csv", periods
        )
    )

    # Menu for user to choose which plot to see
    while True:
        os.system("cls")  # Clear the terminal
        print("\nSelect the dataset to plot the forecast:")  # Print the menu options
        print("1. Area")
        print("2. Estadisticas Matriculados UNAL")
        print("3. Lugar Nacimiento")
        print("4. Lugar Procedencia")
        print("5. Matriculados Primera Vez")
        print("6. Modalidad")
        print("7. Nacionalidad")
        print("8. Sede")
        print("9. Sexo")
        print("0. Exit")

        dataset_choice = int(
            input("Enter the number of your choice: ")
        )  # Get the user's choice

        if dataset_choice == 0:
            print("Exiting the program.")  # Exit the program if the user chooses 0
            break
        elif dataset_choice == 1:
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
            ]  # Columns for "Area" dataset
            processed_data = area_data
            all_predictions = area_predictions
            future_periods = area_future_periods
        elif dataset_choice == 2:
            columns_to_forecast = [
                "Postgrado",
                "Pregrado",
            ]  # Columns for "Estadisticas Matriculados UNAL" dataset
            processed_data = estadisticas_data
            all_predictions = estadisticas_predictions
            future_periods = estadisticas_future_periods
        elif dataset_choice == 3:
            columns_to_forecast = lugar_nacimiento_data[
                "Departamento"
            ].unique()  # Columns for "Lugar Nacimiento" dataset
            processed_data = lugar_nacimiento_data
            all_predictions = lugar_nacimiento_predictions
            future_periods = lugar_nacimiento_future_periods
        elif dataset_choice == 4:
            columns_to_forecast = lugar_procedencia_data[
                "Departamento"
            ].unique()  # Columns for "Lugar Procedencia" dataset
            processed_data = lugar_procedencia_data
            all_predictions = lugar_procedencia_predictions
            future_periods = lugar_procedencia_future_periods
        elif dataset_choice == 5:
            columns_to_forecast = (
                matriculados_columns  # Columns for "Matriculados Primera Vez" dataset
            )
            processed_data = matriculados_data
            all_predictions = matriculados_predictions
            future_periods = matriculados_future_periods
        elif dataset_choice == 6:
            columns_to_forecast = [
                "Doctorado",
                "Especialidades médicas",
                "Especialización",
                "Maestría",
                "Pregrado",
            ]  # Columns for "Modalidad" dataset
            processed_data = modalidad_data
            all_predictions = modalidad_predictions
            future_periods = modalidad_future_periods
        elif dataset_choice == 7:
            columns_to_forecast = [
                "Colombiana",
                "Extranjero",
                "Sin información",
            ]  # Columns for "Nacionalidad" dataset
            processed_data = nacionalidad_data
            all_predictions = nacionalidad_predictions
            future_periods = nacionalidad_future_periods
        elif dataset_choice == 8:
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
            ]  # Columns for "Sede" dataset
            processed_data = sede_data
            all_predictions = sede_predictions
            future_periods = sede_future_periods
        elif dataset_choice == 9:
            columns_to_forecast = ["Hombres", "Mujeres"]  # Columns for "Sexo" dataset
            processed_data = sexo_data
            all_predictions = sexo_predictions
            future_periods = sexo_future_periods
        else:
            print(
                "Invalid choice. Please try again."
            )  # Print an error message if the user enters an invalid choice
            continue

        while True:
            print(
                "\nSelect the column to plot the forecast:"
            )  # Print the column selection menu
            for i, column in enumerate(columns_to_forecast, 1):
                print(
                    f"{i}. {column}"
                )  # Print each column with its corresponding number
            print(
                f"{len(columns_to_forecast) + 1}. Total"
            )  # Print the option to plot the total forecast
            print(
                "0. Back to dataset selection"
            )  # Print the option to go back to the dataset selection menu

            column_choice = int(
                input("Enter the number of your choice: ")
            )  # Get the user's choice

            if column_choice == 0:
                break  # Go back to the dataset selection menu if the user chooses 0
            elif 1 <= column_choice <= len(columns_to_forecast):
                selected_column = columns_to_forecast[
                    column_choice - 1
                ]  # Get the selected column
                plot_forecast(
                    processed_data,
                    all_predictions[selected_column],
                    future_periods,
                    selected_column,
                )  # Plot the forecast for the selected column
            elif column_choice == len(columns_to_forecast) + 1:
                plot_forecast(
                    processed_data, all_predictions["Total"], future_periods, "Total"
                )  # Plot the total forecast
            else:
                print(
                    "Invalid choice. Please try again."
                )  # Print an error message if the user enters an invalid choice


if __name__ == "__main__":
    main()  # Call the main function to start the program