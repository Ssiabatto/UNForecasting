# UNForecasting

This project aims to forecast student enrollment data for various categories such as area of study, campus, nationality, and more using time series forecasting models. The forecasts are visualized using interactive plots.

## Project Structure

```
data
├── csv
│   ├── Area del Conocimiente.csv       # Contains the knowledge area distribution of the enrolled students
│   ├── Estadisticas Admitidos.csv      # Contains the amount of students accepted on each period, separated by "Pregrado" and "Postgrado"
│   ├── Estadisticas Aspirantes.csv     # Contains the amount of applicants, separated by "Pregrado" and "Postgrado"
│   ├── Estadisticas Matriculados.csv   # Contains the amount of students enrolled on a "Pregrado" and "Postgrado"
│   ├── Estrato.csv                     # Contains the stats about the economic condition of the enrolled students
│   ├── Lugar Nacimiento.csv            # Contains the stats about the birthplaces of the students
│   ├── Lugar Procedencia.csv           # Contains the stats about the locations from where the students come.
│   ├── Matriculados Primera Vez.csv    # Contains the amount of students who enrolled for the first semester and the "old" students
│   ├── Modalidad.csv                   # Contains the info regarding the type of enrollment of the students
│   ├── Nacionalidad.csv                # Contains the stats regarding the country of birth of the students
│   ├── Sede.csv                        # Contains the stats of enrollment on each headquarters
│   └── Sexo.csv                        # Contains the info about the number of Men and Women enrolled
├── excels
│   ├── Area del Conocimiente.xlsx      # Contains the knowledge area distribution of the enrolled students
│   ├── Estadisticas Admitidos.xlsx     # Contains the amount of students accepted on each period, separated by "Pregrado" and "Postgrado"
│   ├── Estadisticas Aspirantes.xlsx    # Contains the amount of applicants, separated by "Pregrado" and "Postgrado"
│   ├── Estadisticas Matriculados.xlsx  # Contains the amount of students enrolled on a "Pregrado" and "Postgrado"
│   ├── Estrato.xlsx                    # Contains the stats about the economic condition of the enrolled students
│   ├── Lugar Nacimiento.xlsx           # Contains the stats about the birthplaces of the students
│   ├── Lugar Procedencia.xlsx          # Contains the stats about the locations from where the students come.
│   ├── Matriculados Primera Vez.xlsx   # Contains the amount of students who enrolled for the first semester and the "old" students
│   ├── Modalidad.xlsx                  # Contains the info regarding the type of enrollment of the students
│   ├── Nacionalidad.xlsx               # Contains the stats regarding the country of birth of the students
│   ├── PorcentajesAdmitidos.xlsx       # Contains the percentage of accepted students in relation to the number of applicants
│   ├── Sede.xlsx                       # Contains the stats of enrollment on each headquarters
│   └── Sexo.xlsx                       # Contains the info about the number of Men and Women enrolled
├── predictions (Basically the same as csv but these are projections, 10 semesters ahead of the last historical data on csv) (These are the current projections, these may change)
│   ├── Area.csv                       # Contains the area distribution of the enrolled students
│   ├── Area.csv                       # Contains the area distribution of the
│   ├── Estadisticas Matriculados.csv  # Contains the amount of students on a "Pregrado" and "Postgrado"
│   ├── Lugar Nacimiento.csv           # Contains the stats about the birthplaces of the students
│   ├── Lugar Procedencia.csv          # Contains the stats about the locations from where the students come.
│   ├── Matriculados Primera Vez.csv   # Contains the amount of students who enrolled for the first semester and the "old" students
│   ├── Modalidad.csv                  # Contains the info regarding the type of enrollment of the students
│   ├── Nacionalidad.csv               # Contains the stats regarding the country of birth of the students
│   ├── Sede.csv                       # Contains the stats of enrollment on each headquarters
│   └── Sexo.csv                       # Contains the info about the number of Men and Women enrolled
├── src
│   ├── analyze_data.py          # Code for analyzing the data and determining parameters for the SARIMAX model
│   ├── data_preprocessing.py    # Code for loading and preprocessing the data
│   ├── forecasting_model.py     # Code for training the SARIMAX model and generating forecasts
│   ├── main.py                  # Main script that orchestrates data loading, forecasting, and visualization
│   └── visualization.py         # Code for plotting the historical data and forecasts
├── requirements.txt      # Lists project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

## Installation

1. Clone the repository:

   ```sh
   git clone <repository_url>
   cd UNForecasting
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:

   ```sh
   python src/main.py
   ```

2. Follow the on-screen instructions to select the dataset and column for which you want to see the forecast.

## Project Files

- **data_preprocessing.py**: Contains functions to load and preprocess the data.
- **forecasting_model.py**: Contains functions to train the SARIMAX model and generate forecasts.
- **visualization.py**: Contains functions to plot the historical data and forecasts.
- **main.py**: The main script that orchestrates the data loading, forecasting, and visualization.
- **analyze_data.py**: Contains functions to analyze the data and determine the parameters for the SARIMAX model.

## Example

To forecast the enrollment data for the "Matriculados Primera Vez" dataset, run the main script and select the corresponding option from the menu. The forecasted data will be saved in the [predictions] directory and an interactive plot will be displayed.

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- statsmodels
- plotly

## License

This project is licensed under the MIT License.
