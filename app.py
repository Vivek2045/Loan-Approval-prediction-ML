import os
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Retrieve form data
            applicant_income = float(request.form["ApplicantIncome"])
            coapplicant_income = float(request.form["CoapplicantIncome"])
            credit_history = int(request.form["Credit_History"])
            self_employed = int(request.form["Self_Employed"])
            property_area = int(request.form["Property_Area"])

            # Simple prediction logic (replace this with your model)
            if credit_history == 1 and applicant_income > 3000:
                prediction = "Loan Approved"
            else:
                prediction = "Loan Not Approved"
        except ValueError:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


