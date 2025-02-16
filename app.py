from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

app.secret_key = 'restaurant_rating_prediction'  # Required for sessions

# Load ML model and scaler
scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/restaurant_rating_predictor.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get user inputs from the form
            average_cost = float(request.form["average_cost"])
            table_booking = 1 if request.form["table_booking"] == "Yes" else 0
            online_delivery = 1 if request.form["online_delivery"] == "Yes" else 0
            price_range = int(request.form["price_range"])

            # Prepare data for the model
            values = np.array([[average_cost, table_booking, online_delivery, price_range]])
            scaled_values = scaler.transform(values)

            # Make prediction
            prediction = model.predict(scaled_values)[0]

            # Determine rating category
            if prediction < 2.5:
                rating = "⭐ Poor"
            elif 2.5 <= prediction < 3.5:
                rating = "⭐⭐ Average"
            elif 3.5 <= prediction < 4.0:
                rating = "⭐⭐⭐ Good"
            elif 4.0 <= prediction < 4.5:
                rating = "⭐⭐⭐⭐ Very Good"
            else:
                rating = "⭐⭐⭐⭐⭐ Excellent"

            return render_template("index.html", rating=rating, score=round(prediction, 2))

        except Exception as e:
            return render_template("index.html", error=f"Error: {e}")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
