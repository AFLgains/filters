import datetime
import pytz
import os
import logging
import numpy as np
import cryptocompare
import pandas as pd
from typing import List, Dict
from filters.filter_types.filters import is_buy
import smtplib, ssl
from types import SimpleNamespace
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from types import SimpleNamespace
from filters.config.constants import (EMAIL_DISCLAIMER, EMAIL_SUBJECT, EMAIL_TICKER,
                                      EMAIL_DATAAGE, EMAIL_NEXTDATA, EMAIL_CURRENTPRICE,
                                      EMAIL_TABLEHEADER,EMAIL_LINE,_EMAIL_TABLE_FORMAT,
                                      EMPTY_PORTFOLIO, FULL_PORTFOLIO,_EMAIL_TABLE_FORMAT_FLOAT)

logger = logging.getLogger(__name__)


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


def set_api_key(api_key):
    cryptocompare.cryptocompare._set_api_key_parameter(api_key)


def end_date(delta=1):
    return datetime.date.today() + datetime.timedelta(days=delta)


def return_price_history_df(ticker,fiat = "USD",limit = 2000):
    download = pd.DataFrame.from_dict(
        cryptocompare.get_historical_price_day(
            ticker,
            fiat,
            limit=limit,
            toTs=end_date(),
        )
    )
    download.loc[:, "date"] = pd.to_datetime(download.loc[:, "time"], unit="s")
    return download


def get_price_history(ticker, api_key):
    set_api_key(api_key)
    price_history_df = return_price_history_df(ticker)
    return SimpleNamespace(
        price_history_list=list(price_history_df["close"]),
        dates=price_history_df["date"],
        price_history_df=price_history_df,
    )

def gen_email_header(price_history_list: List, ticker: str):
    current_time_str, hrs_old, new_data_hrs = get_time_stats()
    email_message = ""
    email_message += EMAIL_TICKER.format(ticker = ticker)
    email_message += EMAIL_SUBJECT.format(current_time_str = current_time_str)
    email_message += EMAIL_DATAAGE.format(hrs = round(hrs_old.seconds / 3600, 2))
    email_message += EMAIL_NEXTDATA.format(hrs = round(new_data_hrs.seconds/ 3600, 2))
    email_message += EMAIL_CURRENTPRICE.format(price = price_history_list[-1])
    email_message += EMAIL_TABLEHEADER
    email_message += EMAIL_LINE
    return email_message

def get_last_pos_vel(strat, price_history_list):
    pos, vel, cov = strat.calc_momentum(price_history=price_history_list)
    return pos[-1], vel[-1], cov[-1]

def buy_signal_out_eth(pos, vel, cov):
    buy_signal, _ = is_buy(pos, vel, cov, FULL_PORTFOLIO)
    return "buy" if buy_signal else "HODL CASH"

def sell_signal_in_eth(pos, vel, cov):
    _, sell_signal = is_buy(pos, vel, cov, EMPTY_PORTFOLIO)
    return "sell" if sell_signal else "HODL ETH"

def gen_email_body(price_history_list, strats):
    email_message=""
    for strat in strats:
        price_stats=get_last_pos_vel(strat, price_history_list)
        signal_if_out = buy_signal_out_eth(*price_stats)
        signal_if_in = sell_signal_in_eth(*price_stats)
        email_message += _EMAIL_TABLE_FORMAT_FLOAT % (
            strat.name,
            price_stats[0],
            price_stats[1],
            signal_if_in,
            signal_if_out,
        )
    return email_message

def gen_email_disclaimer():
    return EMAIL_DISCLAIMER

def gen_email_subject():
    current_time_str, _, _ = get_time_stats()
    return EMAIL_SUBJECT.format(current_time_str = current_time_str)

def vis_live_strats(price_history_list: List, strats: List, ticker: str) -> str:
    email_header = gen_email_header(price_history_list, ticker)
    email_body = gen_email_body(price_history_list, strats)
    disclaimer = gen_email_disclaimer()
    email_subject = gen_email_subject()
    email_message = email_header + email_body + disclaimer
    print(email_message)
    pos_estimate = [strat.calc_momentum(price_history=price_history_list)[0] for strat in strats]
    vel_estimate = [strat.calc_momentum(price_history=price_history_list)[1] for strat in strats]
    return SimpleNamespace(pos_estimate = pos_estimate, vel_estimate = vel_estimate),\
           SimpleNamespace(email_message=email_message, email_subject = email_subject)


def send_email(
    output_filename,
    password,
    mailing_list,
    email_content,
    email: bool = False,
):
    message_str = email_content.email_message
    subject_str = email_content.email_subject
    if email:
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


def parse_email_list(email_list):
    return os.environ.get("MAILING_LIST").split(";") or email_list.split(";")
