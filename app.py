from flask import Flask, render_template, request, session
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib
import socket
import os
import random
import traceback


# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "fraudshield_demo_secret_key"
)

load_dotenv()


# ==================================================
# GMAIL CONFIGURATION
# ==================================================

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD")

if not SENDER_EMAIL or not SENDER_APP_PASSWORD:
    print("WARNING: Gmail configuration is missing.")
else:
    print("Gmail configuration loaded successfully.")


# ==================================================
# LOAD FRAUDSHIELD MODEL
# ==================================================

model_package = joblib.load(
    "model/fraud_model.pkl"
)

model = model_package["model"]
features = model_package["features"]
median_amount = model_package["median_amount"]

print("Model loaded successfully!")
print("Model features:", features)


# ==================================================
# DEMO BALANCE
# ==================================================

DEMO_BALANCE = 10000.00


# ==================================================
# SEND OTP THROUGH GMAIL
# ==================================================

def send_otp_email(
    receiver_email,
    otp,
    withdrawal_amount
):

    try:

        message = EmailMessage()

        message["Subject"] = (
            "FraudShield Transaction Verification OTP"
        )

        message["From"] = SENDER_EMAIL

        message["To"] = receiver_email

        message.set_content(
            f"""
FraudShield Transaction Verification

Your OTP for the transaction is:

{otp}

Withdrawal Amount: ₹{withdrawal_amount:.2f}

This OTP is required to complete your transaction.

Please do not share this OTP with anyone.

Regards,
FraudShield Security System
"""
        )

        print("----------------------------------------")
        print("Preparing Gmail OTP...")
        print("Receiver:", receiver_email)
        print("----------------------------------------")

        # Force IPv4 for Gmail SMTP
        original_getaddrinfo = socket.getaddrinfo

        def ipv4_getaddrinfo(
            host,
            port,
            family=0,
            type=0,
            proto=0,
            flags=0
        ):

            return original_getaddrinfo(
                host,
                port,
                socket.AF_INET,
                type,
                proto,
                flags
            )

        socket.getaddrinfo = ipv4_getaddrinfo

        try:

            print("Connecting to Gmail using IPv4...")

            with smtplib.SMTP_SSL(
                "smtp.gmail.com",
                465,
                timeout=30
            ) as smtp:

                print(
                    "Gmail SMTP connection successful."
                )

                smtp.login(
                    SENDER_EMAIL,
                    SENDER_APP_PASSWORD
                )

                print(
                    "Gmail login successful."
                )

                smtp.send_message(message)

                print(
                    "OTP email sent successfully."
                )

        finally:

            socket.getaddrinfo = original_getaddrinfo

        return True

    except Exception as e:

        print("----------------------------------------")
        print("EMAIL SENDING ERROR:")
        print(repr(e))
        print("----------------------------------------")

        return False


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():

    balance = session.get(
        "balance",
        DEMO_BALANCE
    )

    return render_template(
        "index.html",
        balance=balance
    )


# ==================================================
# FRAUD PREDICTION
# ==================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ------------------------------------------
        # GET FORM DATA
        # ------------------------------------------

        transaction = request.form.to_dict()


        # ------------------------------------------
        # GET GMAIL ADDRESS
        # ------------------------------------------

        receiver_email = transaction.get(
            "email",
            ""
        ).strip()


        if not receiver_email:

            return render_template(
                "index.html",
                balance=session.get(
                    "balance",
                    DEMO_BALANCE
                ),
                error=(
                    "Please enter your Gmail address."
                )
            )


        # ------------------------------------------
        # GET TRANSACTION HOUR
        # ------------------------------------------

        transaction_hour = int(
            transaction.get(
                "hour",
                transaction.get(
                    "Hour",
                    0
                )
            )
        )


        # ------------------------------------------
        # GET WITHDRAWAL AMOUNT
        # ------------------------------------------

        withdrawal_amount = float(
            transaction.get(
                "Amount",
                0
            )
        )


        # ------------------------------------------
        # VALIDATE HOUR
        # ------------------------------------------

        if (
            transaction_hour < 0
            or transaction_hour > 23
        ):

            return render_template(
                "index.html",
                balance=session.get(
                    "balance",
                    DEMO_BALANCE
                ),
                error=(
                    "Transaction hour must "
                    "be between 0 and 23."
                )
            )


        # ------------------------------------------
        # VALIDATE AMOUNT
        # ------------------------------------------

        if withdrawal_amount <= 0:

            return render_template(
                "index.html",
                balance=session.get(
                    "balance",
                    DEMO_BALANCE
                ),
                error=(
                    "Withdrawal amount must "
                    "be greater than ₹0."
                )
            )


        # ------------------------------------------
        # CHECK DEMO BALANCE
        # ------------------------------------------

        balance = session.get(
            "balance",
            DEMO_BALANCE
        )


        if withdrawal_amount > balance:

            return render_template(
                "index.html",
                balance=balance,
                error=(
                    "Insufficient demo balance."
                )
            )


        # ==================================================
        # CREATE MODEL INPUT FEATURES
        # ==================================================

        # Convert hour into seconds
        transaction_time = (
            transaction_hour * 3600
        )


        # Log transformation of amount
        amount_log = np.log1p(
            withdrawal_amount
        )


        # Amount compared with dataset median
        amount_ratio = (
            withdrawal_amount
            / (median_amount + 0.01)
        )


        # ------------------------------------------
        # CREATE INPUT VALUES
        # ------------------------------------------

        input_values = {

            "Time": transaction_time,

            "Hour": transaction_hour,

            "Amount": withdrawal_amount,

            "Amount_Log": amount_log,

            "Amount_Ratio": amount_ratio
        }


        # ------------------------------------------
        # ARRANGE FEATURES IN MODEL ORDER
        # ------------------------------------------

        input_data = pd.DataFrame(
            [[
                input_values[feature]
                for feature in features
            ]],
            columns=features
        )


        print("----------------------------------------")
        print("Transaction received")
        print("Hour:", transaction_hour)
        print(
            "Amount:",
            withdrawal_amount
        )
        print(
            "Model input:",
            input_data.to_dict(
                orient="records"
            )
        )
        print("----------------------------------------")


        # ==================================================
        # ML PREDICTION
        # ==================================================

        prediction = model.predict(
            input_data
        )[0]


        # ------------------------------------------
        # FRAUD PROBABILITY
        # ------------------------------------------

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = model.predict_proba(
                input_data
            )[0][1]

            probability = round(
                probability * 100,
                2
            )


        # ==================================================
        # RISK LEVEL
        # ==================================================

        if probability is not None:

            if probability < 30:

                risk_level = "LOW RISK"
                risk_icon = "🟢"

            elif probability < 70:

                risk_level = "MEDIUM RISK"
                risk_icon = "🟡"

            else:

                risk_level = "HIGH RISK"
                risk_icon = "🔴"

        else:

            risk_level = "NOT AVAILABLE"
            risk_icon = "⚪"


        # ==================================================
        # TRANSACTION RESULT
        # ==================================================

        if prediction == 1:

            result = (
                "⚠️ Fraudulent Transaction Detected"
            )

        else:

            result = (
                "✅ Legitimate Transaction"
            )


        # ==================================================
        # BLOCK HIGH-RISK TRANSACTION
        # ==================================================

        if (
            probability is not None
            and probability >= 70
        ):

            return render_template(
                "index.html",
                balance=balance,
                prediction=result,
                probability=probability,
                risk_level=risk_level,
                risk_icon=risk_icon,
                blocked=True
            )


        # ==================================================
        # GENERATE OTP
        # ==================================================

        otp = str(
            random.randint(
                100000,
                999999
            )
        )


        session["otp"] = otp

        session["pending_amount"] = (
            withdrawal_amount
        )

        session["receiver_email"] = (
            receiver_email
        )


        # ==================================================
        # SEND OTP
        # ==================================================

        email_sent = send_otp_email(
            receiver_email,
            otp,
            withdrawal_amount
        )


        if not email_sent:

            return render_template(
                "index.html",
                balance=balance,
                prediction=result,
                probability=probability,
                risk_level=risk_level,
                risk_icon=risk_icon,
                error=(
                    "Unable to send OTP. "
                    "Please check email configuration."
                )
            )


        # ==================================================
        # OTP REQUIRED
        # ==================================================

        return render_template(
            "index.html",
            balance=balance,
            prediction=result,
            probability=probability,
            risk_level=risk_level,
            risk_icon=risk_icon,
            otp_required=True,
            withdrawal_amount=withdrawal_amount
        )


    except Exception as e:

        print("----------------------------------------")
        print("PREDICTION ERROR:")
        print(repr(e))
        traceback.print_exc()
        print("----------------------------------------")


        return render_template(
            "index.html",
            balance=session.get(
                "balance",
                DEMO_BALANCE
            ),
            error=f"Error: {str(e)}"
        )


# ==================================================
# VERIFY OTP
# ==================================================

@app.route(
    "/verify-otp",
    methods=["POST"]
)
def verify_otp():

    entered_otp = request.form.get(
        "otp",
        ""
    ).strip()


    stored_otp = session.get(
        "otp"
    )


    pending_amount = session.get(
        "pending_amount"
    )


    balance = session.get(
        "balance",
        DEMO_BALANCE
    )


    # ------------------------------------------
    # CHECK OTP
    # ------------------------------------------

    if entered_otp == stored_otp:

        if pending_amount is None:

            return render_template(
                "index.html",
                balance=balance,
                error=(
                    "No pending transaction found."
                )
            )


        # --------------------------------------
        # DEDUCT WITHDRAWAL AMOUNT
        # --------------------------------------

        new_balance = (
            balance - pending_amount
        )


        session["balance"] = new_balance


        # --------------------------------------
        # CLEAR PENDING TRANSACTION
        # --------------------------------------

        session.pop(
            "otp",
            None
        )

        session.pop(
            "pending_amount",
            None
        )

        session.pop(
            "receiver_email",
            None
        )


        # --------------------------------------
        # SUCCESS
        # --------------------------------------

        return render_template(
            "index.html",
            balance=new_balance,
            transaction_success=True,
            withdrawn_amount=pending_amount
        )


    # ==================================================
    # WRONG OTP
    # ==================================================

    else:

        return render_template(
            "index.html",
            balance=balance,
            transaction_blocked=True
        )


# ==================================================
# RESET DEMO BALANCE
# ==================================================

@app.route("/reset")
def reset():

    session.clear()

    return render_template(
        "index.html",
        balance=DEMO_BALANCE,
        reset_success=True
    )


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run()