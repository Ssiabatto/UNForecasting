import pandas as pd  # Import pandas for data manipulation


def load_data(filepath):
    data = pd.read_csv(filepath)  # Load the data from the specified CSV file
    return data  # Return the loaded data


def preprocess_data(data):
    # Sort the data by 'Año' and 'Semestre'
    data = data.sort_values(
        by=["Año", "Semestre"]
    )  # Sort the dataframe by year and semester

    # Combine 'Año' and 'Semestre' into a single 'Period' column
    data["Period"] = (
        data["Año"].astype(str) + "-" + data["Semestre"].astype(str)
    )  # Create a 'Period' column by combining 'Año' and 'Semestre'
    data = data.drop(
        columns=["Año", "Semestre"]
    )  # Drop the original 'Año' and 'Semestre' columns

    # Calculate the total if the columns exist
    if all(
        col in data.columns for col in ["Colombiana", "Extranjero", "Sin información"]
    ):  # Check if the specific columns exist
        data["Total"] = data[["Colombiana", "Extranjero", "Sin información"]].sum(
            axis=1
        )  # Calculate the total for these columns
    elif "Total" not in data.columns:  # Check if the 'Total' column does not exist
        data["Total"] = data.sum(axis=1)  # Calculate the total for all columns

    return data  # Return the preprocessed data
