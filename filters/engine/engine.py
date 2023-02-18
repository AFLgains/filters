# TO DO:
# ====================
# [x] Add in trade cost
# [x] Clean up and modularise the code
# [x] Add in other strategies (moving average crossing)
# [x] Add new visualisation code
# [x] add in "perfect" baseline from hodrick prescot
# [x] Add in actual equations of motion - Several candidates - we are using model 2
# [x] Generic optimisattion
# [x] Optimise for mean square error of the velocity
# [x] PnL / drawdown visualisation
# [x] back test on other crypto currencies
# [x] Add a benchmark buy and hold strat
# [x] Add in trading cost
# [x] Set up live velocity tracker
# [x] Clean up code
# [X] Build RTSI Filter
# [x] Research how to tune the algorithm
# [x] Filter that includes a velocity measurement
# [x] Clean up
# [x] Include a scheduler
# [x] Dockerise
# [x] Set up an Email client
# [x] Put onto AWS
# [x] Environment variable
# [x] add email attatchements
# [.] productionise
# [.] Bayesian kalman filter in Stan

# -*- coding: utf-8 -*-
import logging
import pandas as pd
import time
import schedule
import os

from filters.utils.config import Config
from filters.utils.timing import timing
from filters.utils.engine_utils import (
    get_best_params_kf,
    get_best_params_kf3,
    get_price_history,
    vis_live_strats,
    send_email,
)
from filters.filter_types.filters import (
    moving_average,
    diff_moving_average,
    lanczos,
    hodrick_prescot,
    kf,
    kf3,
    kf_velocity,
    buy_hold,
    rsi,
    kalman_filter_package,
)


from filters.utils import vis_utils


class Engine:

    config = None  # type: Config
    log = None

    def __init__(self, config: Config):
        self.config = config
        self.log = logging.getLogger(__name__)
        self.system = os.name

        # Get envirionment varibale
        RECEIVER_EMAIL_LIST = [
            "ethtracker1989@gmail.com",
            "jeanette.fung@hotmail.com",
            "ric.porteous1989@gmail.com",
        ]
        self.password = os.environ.get("PASSWORD")
        self.api_key = os.environ.get("API_KEY")
        self.ticker = os.environ.get("TICKER")
        self.mailing_list = RECEIVER_EMAIL_LIST  # TODO: set through environment variable, i.e., os.environ.get("MAILING_LIST")
        self.output_filename = "/app/filters/output/live.png"
        if self.system == "nt":
            self.output_filename = "filters/output/live.png"

    def run(self, mode):
        if mode == "train":
            self.run_train()
        elif mode == "live":
            self.run_live()
        elif mode == "schedule":
            self.run_schedule()
        else:
            self.log.error("Don't know that mode")

    def run_schedule(self):

        self.run_live()

        # Set up run schedule
        time_to_run = "00:01"
        schedule.every().day.at(time_to_run).do(self.run_live)

        while True:
            schedule.run_pending()
            time.sleep(1)

    @timing
    def run_live(self):

        price_history_list, dates, price_history_df = get_price_history(
            self.ticker, self.api_key
        )

        R, P, Q = get_best_params_kf()
        k_filter_opt = kf(stock_price=price_history_list, R=R, P=P, Q=Q, name="kf1")
        k_filter_vel = kf_velocity(
            stock_price=price_history_list, R=R, P=P, Q=Q, name="kfvel"
        )
        kf_package = kalman_filter_package(
            stock_price=price_history_list, R=R, Q=Q, name="kf_package"
        )

        lan_filter =  lanczos(stock_price=price_history_list, n=10, name="lan")
        ma50 = moving_average(stock_price=price_history_list, n=50, name="ma50")
        ma40 = moving_average(stock_price=price_history_list, n=20, name="ma40")
        ma30 = moving_average(stock_price=price_history_list, n=30, name="ma30")
        ma20 = moving_average(stock_price=price_history_list, n=20, name="ma20")
        ma10 = moving_average(stock_price=price_history_list, n=10, name="ma10")
        mad  = diff_moving_average(
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
            price_history_list, strats, self.ticker
        )
        vis_utils.vis_live_price(
            price_history_df,
            dates,
            pos_estimate,
            vel_estimate,
            strats,
            n_days=100,
            ticker=self.ticker,
            output_filename=self.output_filename,
        )

        if self.config.send_email:
            send_email(
                self.output_filename,
                self.password,
                self.mailing_list,
                email_message,
                email_subject,
            )

    @timing
    def run_train(self):

        # Load the price history
        price_history = pd.read_csv("data/LTC_train.csv")
        price_history_list = list(price_history["close"])
        price_history_list = price_history_list

        # Decompose perfectly using a hodrick prescot filter
        true_trend, true_vel = hodrick_prescot(price_history_list, lam=100)

        # Specify the different filters we would like
        # best_params = tune.find_optimal_kalman_filter(price_history_list)
        # best_params = tune.find_optimal_kalman_filter_k3(price_history_list,true_vel)

        R, P, Q = get_best_params_kf()
        k_filter_opt = kf(stock_price=price_history_list, R=R, P=P, Q=Q, name="kf1")
        k_filter_vel = kf_velocity(
            stock_price=price_history_list, R=R, P=P, Q=Q, name="kfvel"
        )

        R, P, Q, H, THETA = get_best_params_kf3()
        k_filter3 = kf3(
            stock_price=price_history_list, R=R, P=P, Q=Q, name="kf3", theta=THETA, H=H
        )

        lan_filter = lanczos(stock_price=price_history_list, n=10, name="lan")
        ma50 = moving_average(stock_price=price_history_list, n=50, name="ma50")
        ma30 = moving_average(stock_price=price_history_list, n=30, name="ma30")
        ma10 = moving_average(stock_price=price_history_list, n=10, name="ma10")
        mad = diff_moving_average(
            stock_price=price_history_list, n1=200, n2=10, name="mad"
        )
        bh = buy_hold(stock_price=price_history_list, name="buy and hold")
        rsi_strat = rsi(stock_price=price_history_list, name="rsi")
        R, P, Q = get_best_params_kf()
        kf_package = kalman_filter_package(
            stock_price=price_history_list, R=R, Q=Q, name="kf_package"
        )

        # Back test all of them
        for strat in [kf_package]:
            res, ex_ante_pos, ex_ante_vel = strat.back_test(
                initial_capital=100, verbose=False, trade_cost=0.005
            )
            print(
                f"{res['name']} - return(anl) = {round(res['annualised_pct_gain'], 2) * 100}% - win rate = {100 * round(res['win_rate'], 2)}% - n_trades = {res['total_trades']}"
            )
            vis_utils.visualise_buy_sells(
                res["name"], price_history_list, true_vel, ex_ante_pos, ex_ante_vel, res
            )
