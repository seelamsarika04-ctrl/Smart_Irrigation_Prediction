from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask application
app = Flask(__name__)

# Load trained ML model
model = joblib.load("irrigation_model.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/dashboard")
def dashboard():

    df = pd.read_csv("dataset/irrigation_labeled.csv")

    total_records = len(df)

    irrigation_counts = df["irrigation_needed"].value_counts()

    required = int(irrigation_counts.get(1, 0))
    not_required = int(irrigation_counts.get(0, 0))

    avg_temperature = round(df["temperature [_ c]"].mean(), 2)

    avg_humidity = round(df["humidity [%]"].mean(), 2)

    avg_soil_moisture = round(df["soil moisture"].mean(), 2)
    # Soil moisture distribution
    moisture_bins = pd.cut(
    df["soil moisture"],
    bins=[-float("inf"), 200, 300, 400, 500, float("inf")],
    labels=["Below 200", "200-299", "300-399", "400-499", "500+"]
    )

    moisture_counts = moisture_bins.value_counts().sort_index()

    moisture_labels = moisture_counts.index.astype(str).tolist()
    moisture_values = moisture_counts.tolist()
    

    return render_template(
        "dashboard.html",
        total_records=total_records,
        required=required,
        not_required=not_required,
        avg_temperature=avg_temperature,
        avg_humidity=avg_humidity,
        avg_soil_moisture=avg_soil_moisture,
        moisture_labels=moisture_labels,
        moisture_values=moisture_values
    )


# Prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Get values from HTML form
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        soil_moisture = float(request.form["soil_moisture"])

        reference_evapotranspiration = float(
          request.form["reference_evapotranspiration"]
        )

        evapotranspiration = float(
          request.form["evapotranspiration"]
        )

        # Prepare input for ML model
        input_data = pd.DataFrame([[
         temperature,
         humidity,
         soil_moisture,
         reference_evapotranspiration,
         evapotranspiration
        ]], columns=[
            "temperature [_ c]",
            "humidity [%]",
            "soil moisture",
            "reference evapotranspiration",
            "evapotranspiration"
        ]) 

        # Predict irrigation
        prediction = model.predict(input_data)[0]

        # Convert prediction into readable result
        if prediction == 1:
            result = "Irrigation Required"
        else:
            result = "Irrigation Not Required"

        return render_template(
            "result.html",
            result=result,
            temperature=temperature,
            humidity=humidity,
            soil_moisture=soil_moisture,
            reference_evapotranspiration=reference_evapotranspiration,    
            evapotranspiration=evapotranspiration
        )

    return render_template("predict.html")


# Run website
if __name__ == "__main__":
    app.run(debug=True)