from flask import Flask, request, render_template
import pandas as pd
import datetime
import joblib

app = Flask(__name__)

supported_cities = {
    "WINDHOEK": "windhoek_model.pkl",
    "NDJAMENA": "ndjamena_model.pkl",
    "NIAMEY AERO": "niamey_aero_model.pkl",
    "TEJGAON": "tejgaon_model.pkl"
}

def repeated(city):
    model = joblib.load(supported_cities[city])

    # load and clean the data
    df = pd.read_csv("weather2.csv", parse_dates=["DATE"])
    df["LOCATION"] = df["NAME"].apply(lambda x: x.split(",")[0].strip())
    df = df[df["LOCATION"].str.upper() == city]
    df.drop(columns=["STATION", "NAME"], inplace=True)

    for col in ["PRCP", "TMAX", "TMIN"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TMAX"] = (df["TMAX"] - 32) * 5/9
    df["TMIN"] = (df["TMIN"] - 32) * 5/9
    df.dropna(subset=["TMAX", "TMIN"], inplace=True)
    df.set_index("DATE", inplace=True)

    # main dataframe
    main_weather = df[["PRCP", "TMAX", "TMIN"]].copy()
    main_weather.columns = ["precipitation", "temperature_max", "temperature_min"]
    main_weather = main_weather.reset_index()
    main_weather = main_weather.sort_values("DATE")
    main_weather["precipitation_lag1"] = main_weather["precipitation"].shift(1)
    main_weather["temperature_max_lag1"] = main_weather["temperature_max"].shift(1)
    main_weather["temperature_min_lag1"] = main_weather["temperature_min"].shift(1)
    main_weather.dropna(inplace=True)

    # select features
    features = ['precipitation', 'temperature_min', 'precipitation_lag1',
                'temperature_max_lag1', 'temperature_min_lag1']

    return model, main_weather, features

# home page
@app.route("/")
def home():
    return render_template('home.html')

# tcurrent weather page
@app.route("/current_weather", methods=["POST"])
def current_weather():
    city = request.form["city"].strip().upper()
    if city not in supported_cities:
        return f"<p>City '{city}' not supported.</p>"

    # get the processed data from the repeated function
    model, main_weather, features = repeated(city)

    # predict for the most recent row
    latest = main_weather.tail(1)
    input_df = latest[features]
    pred = model.predict(input_df)[0]

    pred_max = round(pred[0], 2)
    pred_min = round(pred[1], 2)
    pred_precip = round(pred[2], 2)

    return render_template('current_weather.html', city=city, pred_max=pred_max, pred_min=pred_min, pred_precip=pred_precip)

# 7-day forecast
@app.route("/forecast", methods=["POST"])
def forecast():
    city = request.form["city"].strip().upper()
    if city not in supported_cities:
        return f"<p>City '{city}' not supported.</p>"

    # get the processed data from the repeated function
    model, main_weather, features = repeated(city)

    # get last 7 days for forecasting
    recent_days = main_weather.tail(7)
    future_dates = [(datetime.datetime.today() + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    predictions = []

    for i in range(min(7, len(recent_days))):
        row = recent_days.iloc[i][features]
        input_df = pd.DataFrame([row])
        result = model.predict(input_df)[0]
        temp_max, temp_min, rain = [round(val, 2) for val in result]
        predictions.append([future_dates[i], temp_max, temp_min, rain])

    # label extreme weather
    labels = []
    heat = flood = 0
    safety_links = {}

    for row in predictions:
        label = []
        date, tmax, tmin, rain = row

        if tmax >= 35:
            heat += 1
        else:
            heat = 0
        if heat >= 3:
            label.append("Heat Wave")
            safety_links["Heat Wave"] = "https://www.nhs.uk/live-well/seasonal-health/heatwave-how-to-cope-in-hot-weather/"

        if rain >= 20:
            flood += 1
        else:
            flood = 0
        if flood >= 1:
            label.append("Possibility of Flash Flood")
            safety_links["Flash Flood Risk"] = "https://www.gov.uk/help-during-flood"

        if rain >= 48:
            label.append("Heavy Rain")
            safety_links["Extreme Rain"] = "https://weather.metoffice.gov.uk/warnings-and-advice/seasonal-advice/stay-safe-in-heavy-rain"

        if tmax < 0:
            label.append("Extreme Cold")
            safety_links["Extreme Cold"] = "https://www.weather.gov/safety/cold-during"

        labels.append("No extreme weather" if not label else " and ".join(label))

    # make output
    output = []
    for i in range(len(predictions)):
        output.append({
            'date': predictions[i][0],
            'max_temp': predictions[i][1],
            'min_temp': predictions[i][2],
            'rain': predictions[i][3],
            'alert': labels[i]
        })

    return render_template('forecast.html', city=city, forecast=output, safety_links=safety_links)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
