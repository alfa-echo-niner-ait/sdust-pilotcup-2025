# Imports and Setup
import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize the Dash App and Load Assets
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Air Pollution Analysis & Prediction"

# Load data and ML models
try:
    df = pd.read_csv("dataset/modified_dataframe.csv", parse_dates=["Date"])

    model = load_model("model_files/unified_lstm_model.h5")
    model.compile(optimizer="adam", loss="mean_squared_error")  # Suppress warning

    pm25_scaler = joblib.load("model_files/unified_pm25_scaler.pkl")
    scaler = joblib.load("model_files/full_feature_scaler.pkl")  # feature scaler
    city_encoder = joblib.load("model_files/city_encoder.pkl")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")

all_cities = df["City"].unique()
all_species = df["Specie"].unique()
time_step = 60

# Define App Layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Air Pollution Analysis & Prediction",
                            className="text-primary-emphasis",
                        ),
                        html.P(
                            "An interactive dashboard to explore historical air quality data and predict future PM2.5 levels.",
                            className="lead",
                        ),
                    ],
                    width=12,
                )
            ],
            className="my-4 text-center",
        ),
        # Data Exploration Section
        dbc.Card(
            [
                dbc.CardHeader(
                    "Historical Data Explorer",
                    className="bg-primary-subtle text-primary-emphasis",
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="city-dropdown",
                                        options=[
                                            {"label": city, "value": city}
                                            for city in all_cities
                                        ],
                                        value=["Beijing"],
                                        multi=True,
                                    ),
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="specie-dropdown",
                                        options=[
                                            {"label": specie, "value": specie}
                                            for specie in all_species
                                        ],
                                        value="pm25",
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dcc.DatePickerRange(
                                        id="date-picker-range",
                                        min_date_allowed=df["Date"].min().date(),
                                        max_date_allowed=df["Date"].max().date(),
                                        initial_visible_month=df["Date"].max().date(),
                                        start_date=(
                                            df["Date"].max() - timedelta(days=365)
                                        ).date(),
                                        end_date=df["Date"].max().date(),
                                        display_format="YYYY-MM-DD",
                                    ),
                                    width=5,
                                ),
                            ]
                        ),
                        dcc.Loading(dcc.Graph(id="historical-graph"), type="cube"),
                    ]
                ),
            ],
            className="mb-4",
        ),
        # Prediction Section
        dbc.Card(
            [
                dbc.CardHeader(
                    "PM2.5 Prediction Engine",
                    className="bg-success-subtle text-success-emphasis",
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Select City to Predict:"),
                                        dcc.Dropdown(
                                            id="predict-city-dropdown",
                                            options=[
                                                {"label": city, "value": city}
                                                for city in all_cities
                                            ],
                                            value="Beijing",
                                            clearable=False,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Forecast Horizon:"),
                                        dcc.Dropdown(
                                            id="forecast-horizon",
                                            options=[
                                                {"label": "1 day", "value": 1},
                                                {"label": "3 days", "value": 3},
                                                {"label": "7 days", "value": 7},
                                            ],
                                            value=7,
                                            clearable=False,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Click to Forecast:"),
                                        html.Br(),
                                        dbc.Button(
                                            "Predict PM2.5",
                                            id="predict-button",
                                            n_clicks=0,
                                            color="success",
                                        ),
                                    ],
                                    width=4,
                                    className="mt-auto",
                                ),
                            ]
                        ),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(id="prediction-output", type="cube"),
                                    width=12,
                                )
                            ]
                        ),
                    ]
                ),
            ],
            className="mb-4",
        ),
        # Show the saved Folium Map
        dbc.Card(
            [
                dbc.CardHeader(
                    "China PM2.5 Geospatial Map",
                    className="bg-info-subtle text-info-emphasis",
                ),
                dbc.CardBody(
                    [
                        html.Iframe(
                            id="folium-map",
                            srcDoc=(
                                open(
                                    "assets/china_air_quality_map.html",
                                    "r",
                                    encoding="utf-8",
                                ).read()
                                if os.path.exists("assets/china_air_quality_map.html")
                                else "<h4>Map not found. Please generate and copy to the app root folder china_air_quality_map.html</h4>"
                            ),
                            width="100%",
                            height="600",
                            style={"border": "none"},
                        )
                    ]
                ),
            ],
            className="mb-4",
        ),
        # Footer Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(className="my-4"),
                        html.Div(
                            [
                                html.P(
                                    "Team Members",
                                    className="mb-1 fw-bold fs-5 text-dark",
                                ),
                                html.Div(
                                    [
                                        html.Span("Emon (emon@sdust.edu.cn)"),
                                        html.Span(
                                            "Phonesamay(phonesamay@sdust.edu.cn)"
                                        ),
                                        html.Span("Yu Jinling (jlyu0201@163.com)"),
                                        html.Span("Fu Wenting (fuwenting04@163.com)"),
                                    ],
                                    className="d-flex justify-content-around mb-3",
                                ),
                                html.P(
                                    "© 2025 Team NEXUS. All rights reserved.",
                                    className="footer bg-dark text-light mt-3 p-3",
                                ),
                            ],
                            className="text-center",
                        ),
                    ],
                    width=12,
                )
            ],
            className="mt-5",
        ),
    ],
    fluid=True,
    className="dbc",
)

# Define Callbacks for Interactivity

# Callback 1: Update Historical Data Graph
@app.callback(
    Output("historical-graph", "figure"),
    [
        Input("city-dropdown", "value"),
        Input("specie-dropdown", "value"),
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
    ],
)
def update_historical_graph(selected_cities, selected_specie, start_date, end_date):
    if not selected_cities:
        return go.Figure().update_layout(
            title_text="Please select at least one city.",
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
        )

    # Convert string dates to datetime objects for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (
        df["City"].isin(selected_cities)
        & (df["Specie"] == selected_specie)
        & (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    )
    filtered_df = df.loc[mask].sort_values("Date")

    if filtered_df.empty:
        return go.Figure().update_layout(
            title_text="No data available for the selected filters",
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
        )

    # Create figure with multiple metrics
    fig = go.Figure()

    # For each city, add multiple traces
    for city in filtered_df["City"].unique():
        city_df = filtered_df[filtered_df["City"] == city]

        # Median line
        fig.add_trace(
            go.Scatter(
                x=city_df["Date"],
                y=city_df["median"],
                mode="lines+markers",
                name=f"{city} - Median",
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Median: %{y:.1f} µg/m³<extra></extra>",
            )
        )

        # Min-Max range
        fig.add_trace(
            go.Scatter(
                x=city_df["Date"],
                y=city_df["max"],
                fill=None,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=city_df["Date"],
                y=city_df["min"],
                fill="tonexty",
                mode="lines",
                name=f"{city} - Range",
                line=dict(width=0),
                fillcolor="rgba(100, 100, 100, 0.2)",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Min: %{y:.1f} µg/m³<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{selected_specie.upper()} Levels for Selected Cities",
        xaxis_title="Date",
        yaxis_title=f"{selected_specie.upper()} Concentration (µg/m³)",
        legend_title="Cities & Metrics",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=50, b=50, t=80),
        height=500,
    )

    # Add range slider for better date navigation
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )

    return fig


# Helper function to categorize air quality
def get_air_quality_level(pm25_value):
    if pm25_value <= 12:
        return "Good"
    elif pm25_value <= 35.4:
        return "Moderate"
    elif pm25_value <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25_value <= 150.4:
        return "Unhealthy"
    elif pm25_value <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# Callback 2: Handle Prediction Logic for Multiple Days
@app.callback(
    Output("prediction-output", "children"),
    [Input("predict-button", "n_clicks")],
    [State("predict-city-dropdown", "value"), State("forecast-horizon", "value")],
)
def update_prediction_output(n_clicks, selected_city, forecast_days):
    if n_clicks == 0:
        return html.Div(
            "Select a city and click the predict button to see the forecast.",
            className="text-center text-muted mt-3",
        )

    try:
        city_df = df[(df["City"] == selected_city) & (df["Specie"] == "pm25")].sort_values(by="Date")
        if len(city_df) < time_step:
            return dbc.Alert(
                f"Not enough data for {selected_city} to make a prediction (need {time_step} days, have {len(city_df)}).",
                color="danger",
            )

        recent_df = city_df.tail(time_step).copy()
        recent_df["year"] = pd.to_numeric(recent_df["Date"].dt.year, errors='coerce')
        recent_df["month"] = pd.to_numeric(recent_df["Date"].dt.month, errors='coerce')
        recent_df["median"] = pd.to_numeric(recent_df["median"], errors='coerce')
        recent_df = recent_df.dropna(subset=["median", "year", "month"])
        recent_df["season"] = recent_df["month"].map(
            lambda m: "Spring" if 3 <= m <= 5 else "Summer" if 6 <= m <= 8 else "Autumn" if 9 <= m <= 11 else "Winter"
        )

        # Scale numeric features
        recent_df[["median", "year", "month"]] = scaler.transform(recent_df[["median", "year", "month"]])
        city_encoded = city_encoder.transform(pd.DataFrame([[selected_city]], columns=["City"]))
        city_encoded_tiled = np.tile(city_encoded, (time_step, 1))

        season_encoded_df = pd.get_dummies(recent_df["season"], prefix='season')
        for season_col in ["season_Autumn", "season_Spring", "season_Summer", "season_Winter"]:
            if season_col not in season_encoded_df.columns:
                season_encoded_df[season_col] = 0
        season_encoded_df = season_encoded_df[["season_Autumn", "season_Spring", "season_Summer", "season_Winter"]].values

        model_input_data = np.hstack([
            recent_df[["median", "year", "month"]].values,
            city_encoded_tiled,
            season_encoded_df
        ]).astype(np.float32)

        predictions = []
        prediction_dates = []
        current_input = model_input_data.copy()

        for day in range(forecast_days):
            input_sequence = current_input[-time_step:]
            model_input = np.reshape(input_sequence, (1, time_step, model_input_data.shape[1]))
            predicted_scaled = model.predict(model_input)
            prediction = pm25_scaler.inverse_transform(predicted_scaled)[0][0]
            predictions.append(prediction)

            last_date = recent_df["Date"].iloc[-1] if not prediction_dates else prediction_dates[-1]
            prediction_date = last_date + timedelta(days=1)
            prediction_dates.append(prediction_date)

            next_year = prediction_date.year
            next_month = prediction_date.month
            next_season = "Spring" if 3 <= next_month <= 5 else "Summer" if 6 <= next_month <= 8 else "Autumn" if 9 <= next_month <= 11 else "Winter"

            new_input_row = np.zeros(model_input_data.shape[1])
            new_input_row[:3] = scaler.transform([[prediction, next_year, next_month]])[0]
            new_input_row[3:3+city_encoded.shape[1]] = city_encoded[0]
            season_index = ["Autumn", "Spring", "Summer", "Winter"].index(next_season)
            new_input_row[3+city_encoded.shape[1]+season_index] = 1
            new_input_row = new_input_row.astype(np.float32)
            current_input = np.vstack([current_input, new_input_row])

        history_dates = city_df["Date"].tail(time_step)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_dates, y=city_df["median"].tail(time_step), mode="lines+markers",
                                 name="Historical PM2.5", line=dict(width=2, color="blue"), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode="lines+markers",
                                 name="Predicted PM2.5", line=dict(width=2, color="red", dash="dot"),
                                 marker=dict(size=8, color="red", symbol="diamond")))

        historical_std = np.std(city_df["median"].tail(30))
        fig.add_trace(go.Scatter(x=prediction_dates + prediction_dates[::-1],
                                 y=[p + historical_std for p in predictions] + [p - historical_std for p in predictions][::-1],
                                 fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,255,255,0)"),
                                 hoverinfo="skip", name="Confidence Interval"))

        fig.update_layout(title=f"{forecast_days}-Day PM2.5 Forecast for {selected_city}",
                          xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)", template="plotly_white",
                          hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                             xanchor="right", x=1))

        prediction_table = html.Div([
            html.H5("Forecast Summary", className="mt-4 mb-3"),
            dash_table.DataTable(
                columns=[{"name": "Date", "id": "date"},
                         {"name": "Predicted PM2.5 (µg/m³)", "id": "pm25"},
                         {"name": "Air Quality", "id": "quality"}],
                data=[{"date": d.strftime("%Y-%m-%d"), "pm25": f"{p:.1f}", "quality": get_air_quality_level(p)}
                      for d, p in zip(prediction_dates, predictions)],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"},
                style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"})
        ])

        return html.Div([dbc.Alert(f"Generated {forecast_days}-day PM2.5 forecast for {selected_city}",
                                   color="success", className="mt-3"),
                         dcc.Graph(figure=fig),
                         prediction_table])

    except Exception as e:
        return dbc.Alert(f"An error occurred during prediction: {str(e)}", color="danger")

# Run the App
if __name__ == "__main__":
    app.run(debug=True)
