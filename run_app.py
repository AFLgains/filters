

import streamlit as st

from filters.utils.engine_utils import (
    get_best_params_kf,
    get_price_history,
    vis_live_strats,
)
from filters.filter_types.filters import (
    moving_average,
    diff_moving_average,
    lanczos,
    kf,
    kf_velocity,
    kalman_filter_package,
)

TICKER = "ETH"
#API_KEY = "6e89c7206509df377432f33c9359bd07e11cc556b3c0976e4107336a648f4460"
API_KEY = "75fbc1adcdfc8a51b20340be281ec2c3e9607ff7bc1ac57c4b3c8f58b61bce58"
def run_eth_tracker():
    price_history_list, dates, price_history_df = get_price_history(
        TICKER, API_KEY
    )

    R, P, Q = get_best_params_kf()
    k_filter_opt = kf(stock_price=price_history_list, R=R, P=P, Q=Q, name="kf1")
    k_filter_vel = kf_velocity(
        stock_price=price_history_list, R=R, P=P, Q=Q, name="kfvel"
    )
    kf_package = kalman_filter_package(
        stock_price=price_history_list, R=R, Q=Q, name="kf_package"
    )

    lan_filter = lanczos(stock_price=price_history_list, n=10, name="lan")
    ma50 = moving_average(stock_price=price_history_list, n=50, name="ma50")
    ma40 = moving_average(stock_price=price_history_list, n=20, name="ma40")
    ma30 = moving_average(stock_price=price_history_list, n=30, name="ma30")
    ma20 = moving_average(stock_price=price_history_list, n=20, name="ma20")
    ma10 = moving_average(stock_price=price_history_list, n=10, name="ma10")
    mad = diff_moving_average(
        stock_price=price_history_list, n1=50, n2=20, name="mad"
    )

    strats = [
        k_filter_opt,
        k_filter_vel,
        kf_package,
        ma50,
        ma40,
        ma30,
        ma20,
        ma10,
        mad,
        lan_filter,
    ]

    pos_estimate, vel_estimate, email_message, email_subject = vis_live_strats(
        price_history_list, strats, TICKER
    )
    return email_subject,email_message

st.title('ETH Tracker')

run = st.button('run')

if run:
    title, contents = run_eth_tracker()
    st.text(title)
    st.text(contents)



