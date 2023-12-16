import datetime
import pytz
import numpy as np
import cryptocompare
import pandas as pd
from typing import List, Dict
from filters.filter_types.filters import is_buy
import smtplib, ssl
import email
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import streamlit as st

def get_time_stats():
    current_time = datetime.datetime.now(datetime.timezone.utc)
    last_data_pull = pytz.utc.localize(
        datetime.datetime.combine(current_time.date(), datetime.time(0, 0, 0, 0))
    )
    next_data_pull = last_data_pull + datetime.timedelta(days=1)
    current_time_str = current_time.strftime("%Y-%m-%d T %H:%M:%S.%f %Z")

    hrs_old = current_time - last_data_pull
    new_data_hrs = next_data_pull - current_time

    return current_time_str, hrs_old, new_data_hrs


def get_price_history(ticker, api_key):

    cryptocompare.cryptocompare._set_api_key_parameter(api_key)

    price_history = cryptocompare.get_historical_price_day(
        ticker,
        "USD",
        limit=2000,
        toTs=datetime.date.today() + datetime.timedelta(days=1),
    )

    price_history_df = pd.DataFrame.from_dict(price_history)
    st.write(price_history_df)
    price_history_df["date"] = pd.to_datetime(price_history_df["time"], unit="s")

    price_history_list = list(price_history_df["close"])
    dates = price_history_df["date"]

    return price_history_list, dates, price_history_df


def get_best_params_kf():
    best_params = {
        "target": 0.6363636363636364,
        "params": {
            "R": 8.620806302527626,
            "qxx": -2.801193802955954,
            "qxy": 0.1,
            "qyy": -5.816125425616493,
        },
    }
    R = best_params["params"]["R"]
    qxx = best_params["params"]["qxx"]
    qxy = best_params["params"]["qxy"]
    qyy = best_params["params"]["qyy"]
    Q = np.array([[np.exp(qxx), qxy], [qxy, np.exp(qyy)]])
    P = 10

    return R, P, Q


def get_best_params_kf3():
    best_params = {
        "target": -93.43627695287812,
        "params": {
            "R": 14.384648571446771,
            "h1": 0.1,
            "h2": 0.0,
            "qxx": -2.0,
            "qxy": 0.1,
            "qyy": -7.36893143479458,
            "x1": 0.1,
            "x2": 0.1,
            "x3": 0.1,
        },
    }
    R = best_params["params"]["R"]
    qxx = best_params["params"]["qxx"]
    qxy = best_params["params"]["qxy"]
    qyy = best_params["params"]["qyy"]
    h1 = best_params["params"]["h1"]
    h2 = best_params["params"]["h2"]
    x1 = best_params["params"]["x1"]
    x2 = best_params["params"]["x2"]
    x3 = best_params["params"]["x3"]

    P = 10
    Q = np.array([[np.exp(qxx), qxy], [qxy, np.exp(qyy)]])
    THETA = np.array([[x1, x2], [0, x3]])
    H = np.array([[h1, h2]])

    return R, P, Q, H, THETA


def vis_live_strats(price_history_list: List, strats: List, ticker: str) -> str:
    current_time_str, hrs_old, new_data_hrs = get_time_stats()

    t = len(price_history_list)
    pos_estimate = []
    vel_estimate = []
    email_message = ""
    email_message += f"\nTicker: {ticker}"
    email_message += f"\nCurrent GMT time is: {current_time_str}"
    email_message += f"\nData pulled is {round(hrs_old.seconds / 3600, 2)} hrs old"
    email_message += (
        f"\nNew daily price pulled in {round(new_data_hrs.seconds / 3600, 2)} hrs"
    )
    email_message += f"\nCurrent price: {price_history_list[t - 1]}"
    email_message += "\n%-10s | %-10s | %-10s | %-10s | %-10s" % (
        "Strat",
        "Price",
        "Vel",
        "If in..",
        "If out..(Cash)",
    )
    email_message += "\n------------------------------------------------------------"
    for strat in strats:
        pos, vel, cov = strat.calc_momentum(price_history=price_history_list)
        buy_signal_in_eth, sell_signal_in_eth = is_buy(
            pos[t - 1], vel[t - 1], cov[t - 1], {"current_cash": 0}
        )
        buy_signal_out_eth, sell_signal_out_eth = is_buy(
            pos[t - 1], vel[t - 1], cov[t - 1], {"current_cash": 1}
        )

        if buy_signal_out_eth:
            signal_if_out = "buy"
        else:
            signal_if_out = "HODL cash"

        if sell_signal_in_eth:
            signal_if_in = "sell"
        else:
            signal_if_in = "HODL"

        email_message += "\n%-10s | %-10.1f | %-10.1f | %-10s | %-10s" % (
            strat.name,
            pos[t - 1],
            vel[t - 1],
            signal_if_in,
            signal_if_out,
        )
        pos_estimate.append(pos)
        vel_estimate.append(vel)

    email_message += (
        "\n\nThese emails are for education and entertainment purposes only,"
        " not financial advice. It is not intended as a substitute for professional"
        " financial, legal or tax advice. I am not a financial professional and "
        "am not aware of your personal financial circumstances. I don't recommend "
        "you follow any recommendations provided by the tracker, either in the past "
        "or in the future. In fact, it would be unreasonable to rely on this "
        "information for any purpose whatsoever."
    )

    print(email_message)
    email_subject = f"ETH PRICE VELOCITY UPDATE: {current_time_str}"
    return pos_estimate, vel_estimate, email_message, email_subject


def send_email(
    output_filename, password, mailing_list, message_str=None, subject_str=None
):

    sender_email = "ethtracker1989@gmail.com"

    if message_str is None:
        message_str = """\
        Subject: Hi there

        This message is sent from Python."""

    if subject_str is None:
        subject_str = "Test"

    port = 465  # For SSL

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("ethtracker1989@gmail.com", password)
        for re in mailing_list:
            # Create a multipart message and set headers
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = re
            message["Subject"] = subject_str

            # Add body to email
            message.attach(MIMEText(message_str, "plain"))

            filename = output_filename  # In same directory as script

            # Open PDF file in binary mode
            with open(filename, "rb") as attachment:
                # Add file as application/octet-stream
                # Email client can usually download this automatically as attachment
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )

            # Add attachment to message and convert message to string
            message.attach(part)
            text = message.as_string()

            server.sendmail(sender_email, re, text)
