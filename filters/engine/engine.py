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
from filters.utils.engine_utils import (
    get_price_history,
    vis_live_strats,
    send_email,
)
from filters.filter_types.filter_tune import get_best_params_kf, get_best_params_kf3

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
from filters.utils.engine_utils import parse_email_list
from filters.utils import vis_utils
from filters.config.constants import _BASE_FILE_NAME, PASSWORD, API_KEY, TIME_TO_RUN
from filters.config.directories import directories
from filters.filter_types import register
from filters.filter_types.generator import FiltersGenerator

from dotenv import load_dotenv, find_dotenv

try:
    load_dotenv(dotenv_path=directories.app /'..'/".env" ) #find_dotenv(raise_error_if_not_found=True)
except OSError:
    raise RuntimeError("'.env' file not found.")


class Engine:

    config = None  # type: Config
    log = None

    def __init__(self, config: Config, email_list: str = None):
        self.config = config
        self.ticker = config.ticker
        self.log = logging.getLogger(__name__)
        self.system = os.name
        self.password = os.environ.get(PASSWORD)
        self.api_key = os.environ.get(API_KEY)
        self.mailing_list = parse_email_list(email_list)
        self.output_filename = directories.output_filename / _BASE_FILE_NAME
        self.register = register

    def run(self, mode, email):
        if mode == "train":
            self.run_train()
        elif mode == "live":
            return self.run_live(email)
        elif mode == "schedule":
            self.run_schedule(email)
        else:
            self.log.error("Don't know that mode")
            raise ValueError

    def run_schedule(self, email):
        self.run_live(email)
        # Set up run schedule
        schedule.every().day.at(TIME_TO_RUN).do(self.run_live(email))
        while True:
            schedule.run_pending()
            time.sleep(1)


    def run_live(self, email):

        price_history = get_price_history(self.ticker, self.api_key)
        generator = FiltersGenerator(registry=self.register)
        strats = generator.generate(stock_price_list=price_history.price_history_list)
        estimates, email_content = vis_live_strats(price_history.price_history_list, strats, self.ticker)
        vis_utils.vis_live_price(
            price_history,
            estimates,
            strats,
            n_days=100,
            ticker=self.ticker,
            output_filename=self.output_filename,
        )

        send_email(
            self.output_filename,
            self.password,
            self.mailing_list,
            email_content,
            email,
        )

        return email_content


    def run_train(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(dir_path, "../data/ETH_test.csv")

        # Load the price history
        price_history = pd.read_csv(data_dir)
        price_history_list = list(price_history["close"])

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
        for strat in [kf_package, k_filter3, rsi_strat, mad, ma10,  ma30, ma50, lan_filter]:
            res, ex_ante_pos, ex_ante_vel = strat.back_test(
                initial_capital=100, verbose=False, trade_cost=0.005
            )
            print(
                f"{res['name']} - return(anl) = {round(res['annualised_pct_gain'], 2) * 100}% - win rate = {100 * round(res['win_rate'], 2)}% - n_trades = {res['total_trades']}"
            )
            vis_utils.visualise_buy_sells(
                res["name"], price_history_list, true_vel, ex_ante_pos, ex_ante_vel, res
            )
