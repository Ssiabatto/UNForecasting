import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter


def plot_forecast(historical_data, predictions, future_periods, column, output_dir):
    try:
        # Convert periods for plotting
        def convert_period(period):
            year, semester = period.split(
                "-"
            )  # Split the period into year and semester
            if semester == "1":
                return f"{year}-01"  # Return January for semester 1
            else:
                return f"{year}-02"  # Return July for semester 2

        historical_data["PlotPeriod"] = historical_data["Period"].apply(
            convert_period
        )  # Apply the conversion to historical data periods
        future_plot_periods = [
            convert_period(period) for period in future_periods
        ]  # Apply the conversion to future periods

        # Combine historical and future periods
        all_plot_periods = list(historical_data["PlotPeriod"]) + future_plot_periods
        all_values = list(historical_data[column]) + list(predictions)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            historical_data["PlotPeriod"],
            historical_data[column],
            marker="o",
            label="Historical Data",
        )
        plt.plot(
            all_plot_periods[-(len(predictions) + 1) :],
            all_values[-(len(predictions) + 1) :],
            marker="o",
            linestyle="--",
            label="Forecast",
        )
        plt.title(f"Forecast for the next 10 semesters for {column}")
        plt.xlabel("Year-Semester")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.legend()

        # Format the y-axis to display full numbers
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the output file path
        output_filepath = os.path.join(output_dir, f"{column}_forecast.png")

        # Save the plot as an image file
        plt.savefig(output_filepath)
        print(f"Plot saved to {output_filepath}")

        # Display the plot
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the forecast: {e}")
