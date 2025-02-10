import plotly.graph_objects as go  # Import Plotly for interactive plotting


def plot_forecast(historical_data, predictions, future_periods, column):
    # Convert periods for plotting
    def convert_period(period):
        year, semester = period.split("-")  # Split the period into year and semester
        if semester == "1":
            return f"{year}-01"  # Return January for semester 1
        else:
            return f"{year}-07"  # Return July for semester 2

    historical_data["PlotPeriod"] = historical_data["Period"].apply(
        convert_period
    )  # Apply the conversion to historical data periods
    future_plot_periods = [
        convert_period(period) for period in future_periods
    ]  # Apply the conversion to future periods

    # Create a figure
    fig = go.Figure()  # Initialize a new figure

    # Add historical data trace
    fig.add_trace(
        go.Scatter(
            x=historical_data["PlotPeriod"],  # X-axis data
            y=historical_data[column],  # Y-axis data
            mode="lines+markers",  # Plot mode
            name="Historical Data",  # Trace name
        )
    )

    # Add forecasted data trace starting from the last historical period
    all_plot_periods = (
        list(historical_data["PlotPeriod"]) + future_plot_periods
    )  # Combine historical and future periods
    all_values = list(historical_data[column]) + list(
        predictions
    )  # Combine historical and forecasted values
    fig.add_trace(
        go.Scatter(
            x=all_plot_periods[-(len(predictions) + 1) :],  # X-axis data for forecast
            y=all_values[-(len(predictions) + 1) :],  # Y-axis data for forecast
            mode="lines+markers",  # Plot mode
            name="Forecast",  # Trace name
            line=dict(dash="dash"),  # Line style
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Forecast for the next 10 semesters for {column}",  # Plot title
        xaxis_title="Period",  # X-axis title
        yaxis_title="Value",  # Y-axis title
        xaxis=dict(
            tickmode="array",  # Tick mode
            tickvals=all_plot_periods,  # Tick values
            ticktext=list(historical_data["Period"]) + future_periods,  # Tick text
            tickangle=45,  # Tick angle
        ),
        legend=dict(x=0, y=1),  # Legend position
    )

    # Show the figure
    fig.show()  # Display the plot