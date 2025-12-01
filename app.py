from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("AJ_data.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        try:
            # Collect all inputs (same order as training)
            features = [
                float(request.form.get("distance_from_home")),
                float(request.form.get("distance_from_last_transaction")),
                float(request.form.get("ratio_to_median_purchase_price")),
                int(request.form.get("repeat_retailer")),
                int(request.form.get("used_chip")),
                int(request.form.get("used_pin_number")),
                int(request.form.get("online_order"))
            ]

            # Convert to numpy array
            input_data = np.array(features).reshape(1, -1)

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)[0]

            prediction_text = "⚠️ Fraud Transaction Detected!" if prediction == 1 else "✅ Transaction is Legit."

        except Exception as e:
            prediction_text = f"❌ Error: {str(e)}"

    return render_template("temp.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
