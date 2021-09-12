# Script to implement filters as outlined in:
# Trend Filtering Methods for Momentum Strategies
# We are using a super class called trend_detection

from typing import Tuple
import pandas as pd
from pykalman import KalmanFilter

from filters.utils.kalman_filter_utils import *


def is_buy(pos, vel, cov, portfolio):
    buy_signal = False
    sell_signal = False

    if portfolio["current_cash"] == 0:
        buy_signal = False
        if vel < -0.00:
            sell_signal = True
    else:
        sell_signal = False
        if vel > 0.00:
            buy_signal = True
    return buy_signal, sell_signal


def buy_stock(portfolio, current_price, current_vel, date, trade_cost=0):

    buy_amount = portfolio["current_cash"]
    assert buy_amount <= portfolio["current_cash"]
    portfolio["buy_sells"].append(
        [
            {
                "name": "buy",
                "date": date,
                "amount": buy_amount,
                "price": current_price,
                "vel": current_vel,
                "units": "USD",
            }
        ]
    )
    # Update
    portfolio["current_asset"] += buy_amount * (1 - trade_cost) / current_price
    portfolio["current_cash"] -= buy_amount
    return portfolio


def sell_stock(portfolio, current_price, current_vel, date, trade_cost=0):

    sell_amount = portfolio["current_asset"]
    assert sell_amount <= portfolio["current_asset"]
    portfolio["buy_sells"].append(
        [
            {
                "name": "sell",
                "date": date,
                "amount": sell_amount,
                "price": current_price,
                "vel": current_vel,
                "units": "ETH",
            }
        ]
    )
    # Update
    portfolio["current_asset"] -= sell_amount
    portfolio["current_cash"] += sell_amount * current_price * (1 - trade_cost)
    return portfolio


class trend_detector:
    """
    Super class
    """

    def __init__(
        self, stock_price: List, log_returns: bool = False, name: str = "trend_detector"
    ):
        if log_returns:
            self.signal = np.log(stock_price)
        else:
            self.signal = stock_price
        self.n_data = len(stock_price)
        self.name = name

    def __name__(self):
        return self.name

    def calc_momentum(self, price_history: List):
        pos = price_history
        vel = [-100] * len(price_history)
        cov = [0] * len(price_history)
        return pos, vel, cov

    def back_test(
        self, initial_capital=100, trade_cost: float = 0, verbose: bool = True
    ):

        portfolio = {
            "current_cash": initial_capital,
            "current_asset": 0,
            "buy_sells": [],
        }

        pos_estimate = [0, 0]
        vel_estimate = [0, 0]
        for t in range(3, self.n_data + 1):
            pos, vel, cov = self.calc_momentum(price_history=self.signal[0:t])

            pos_estimate.append(pos[t - 1])
            vel_estimate.append(vel[t - 1])
            buy_signal, sell_signal = is_buy(
                pos[t - 1], vel[t - 1], cov[t - 1], portfolio
            )
            current_price = self.signal[t - 1]
            current_vel = vel[t - 1]
            if buy_signal:
                if verbose:
                    print(f"Buying at {current_price}")
                portfolio = buy_stock(
                    portfolio, current_price, current_vel, t - 1, trade_cost
                )
            elif sell_signal:
                portfolio = sell_stock(
                    portfolio, current_price, current_vel, t - 1, trade_cost
                )
                if verbose:
                    print(
                        f"Selling at {current_price}, portfolio_value {portfolio['current_cash']}"
                    )

        assert len(pos_estimate) == self.n_data

        if portfolio["current_cash"] == 0:
            portfolio = sell_stock(portfolio, self.signal[t - 1], current_vel, t - 1)

        return self.compile_strat_results(
            initial_capital=initial_capital,
            n_days=self.n_data,
            strat_results=(portfolio, pos_estimate, vel_estimate),
        )

    def compile_strat_results(
        self, initial_capital: float, n_days, strat_results: Tuple
    ) -> None:

        results = {}
        results["name"] = self.name
        portfolio = strat_results[0]
        ex_ante_pos = strat_results[1]
        ex_ante_vel = strat_results[2]

        results["total_value"] = portfolio["current_cash"]
        results["pct_gain"] = portfolio["current_cash"] / initial_capital - 1
        results["annualised_pct_gain"] = (
            portfolio["current_cash"] / initial_capital
        ) ** (356 / n_days) - 1

        results["buy_dates"] = [
            (d[0]["date"], d[0]["price"], d[0]["vel"])
            for d in portfolio["buy_sells"]
            if d[0]["name"] == "buy"
        ]
        results["sell_dates"] = [
            (d[0]["date"], d[0]["price"], d[0]["vel"])
            for d in portfolio["buy_sells"]
            if d[0]["name"] == "sell"
        ]

        trade_gains = []
        for i in range(len(results["sell_dates"])):
            trade_gains.append(
                results["sell_dates"][i][1] / results["buy_dates"][i][1] - 1
            )

        results["trade_gains"] = trade_gains
        results["total_trades"] = len(trade_gains)

        results["win_rate"] = 0
        if len(results["trade_gains"]) > 0:
            results["win_rate"] = np.mean([s > 0 for s in trade_gains])

        results["ave_gain_when_win"] = 0
        results["ave_loss_when_lose"] = 0
        if len([s for s in trade_gains if s > 0]) > 0:
            results["ave_gain_when_win"] = np.mean([s for s in trade_gains if s > 0])

        if len([s for s in trade_gains if s < 0]) > 0:
            results["ave_loss_when_lose"] = np.mean([s for s in trade_gains if s < 0])

        results["porfolio_value_history"] = [(1, initial_capital)]
        results["porfolio_value_history"].extend(
            [
                (d[0]["date"], d[0]["price"] * d[0]["amount"])
                for d in portfolio["buy_sells"]
                if d[0]["name"] == "sell"
            ]
        )

        results["monthly_pnl"] = create_monthly_pnl(results["porfolio_value_history"])

        return results, ex_ante_pos, ex_ante_vel


def create_monthly_pnl(porfolio_value_history: List) -> List:

    value_df = pd.DataFrame(porfolio_value_history, columns=["day", "Value"])
    value_df = value_df.set_index("day")
    new_index = range(1, max(value_df.index))
    full_value_df = value_df.reindex(new_index).ffill()
    full_value_df["month"] = np.cumsum(full_value_df.index % 30 == 0)
    grouped = full_value_df.groupby("month").agg(["first", "last"])
    grouped["PnL"] = (
        100
        * (grouped[("Value", "last")] - grouped[("Value", "first")])
        / grouped[("Value", "first")]
    )
    return grouped


class moving_average(trend_detector):
    def __init__(
        self,
        stock_price: List,
        n: int,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        """
        Implementing moving average of the price

        :param price_history: List of prices
        :param n: Order of the moving average
        :return: List of the trend
        """
        super().__init__(stock_price, log_returns, name)
        self.n = n
        assert n < self.n_data

    def kron_del(self, a, b):
        if a == b:
            return 1
        else:
            return 0

    def calc_momentum(self, price_history: List):

        n_data = len(price_history)
        L = (
            1
            / float(self.n)
            * (
                np.tril(np.ones([n_data, n_data]), k=0)
                - np.tril(np.ones([n_data, n_data]), k=-self.n)
            )
        )
        l = np.zeros([n_data, n_data])

        for t in range(n_data):  # Rows
            for i in range(n_data):  # Columns
                if (t - self.n) <= i <= t:
                    l[t, i] = (
                        1
                        / self.n
                        * (self.kron_del(i, t) - self.kron_del(i, (t - self.n)))
                    )

        filtered_signal = list(np.dot(L, np.array([price_history]).T))
        filtered_vel = list(np.dot(l, np.array([price_history]).T))

        pos = filtered_signal
        vel = filtered_vel
        cov = [0] * len(price_history)

        return pos, vel, cov


class diff_moving_average(trend_detector):
    def __init__(
        self,
        stock_price: List,
        n1: int,
        n2: int,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        """ """
        super().__init__(stock_price, log_returns, name)
        assert n1 > n2, "n1 > n2"
        self.n1 = n1
        self.n2 = n2

    def calc_moving_average(self, price_history, n):
        n_data = len(price_history)
        L = (
            1
            / float(n)
            * (
                np.tril(np.ones([n_data, n_data]), k=0)
                - np.tril(np.ones([n_data, n_data]), k=-n)
            )
        )
        return list(np.dot(L, np.array([price_history]).T))

    def calc_momentum(self, price_history: List):

        mu = (
            2
            / (self.n1 - self.n2)
            * (
                np.array(self.calc_moving_average(price_history, self.n2))
                - np.array(self.calc_moving_average(price_history, self.n1))
            )
        )

        pos = self.calc_moving_average(price_history, self.n2)
        vel = mu
        cov = [0] * len(price_history)

        return pos, vel, cov


class asymetric_triangular(trend_detector):
    def __init__(
        self,
        stock_price: List,
        n: int,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        super().__init__(stock_price, log_returns, name)
        self.n = n
        assert n < self.n_data

    def calc_momentum(self, price_history: List):

        L = np.zeros([self.n_data, self.n_data])
        for t in range(self.n_data):  # Rows
            for i in range(self.n_data):  # Columns
                if (t - self.n) <= i <= t:
                    L[t, i] = 2 / self.n ** 2 * (self.n - (t - i))

        filtered_signal = list(np.dot(L, np.array([price_history]).T))

        pos = filtered_signal
        vel = [100] * len(price_history)
        cov = [0] * len(price_history)

        return pos, vel, cov


class lanczos(trend_detector):
    def __init__(
        self,
        stock_price: List,
        n: int,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        super().__init__(stock_price, log_returns, name)
        self.n = n
        assert n < self.n_data

    def calc_momentum(self, price_history: List):

        n_data = len(price_history)

        L = np.zeros([n_data, n_data])
        l = np.zeros([n_data, n_data])
        for t in range(n_data):  # Rows
            for i in range(n_data):  # Columns
                if (t - self.n) <= i <= t:
                    L[t, i] = 6 / self.n ** 3 * (t - i) * (self.n - (t - i))
                    l[t, i] = 12 / self.n ** 3 * (self.n / 2 - (t - i))

        filtered_signal = list(np.dot(L, np.array([price_history]).T))
        filtered_velocity = list(np.dot(l, np.array([price_history]).T))

        pos = filtered_signal
        vel = filtered_velocity
        cov = [0] * len(price_history)

        return pos, vel, cov


class kf(trend_detector):
    def __init__(
        self,
        stock_price: List,
        R,
        Q,
        P,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        super().__init__(stock_price, log_returns, name)
        self.R = R
        self.Q = Q
        self.P = P

    def calc_momentum(self, price_history: List):

        pos, vel, cov = kalman_filter_stock(self.R, self.Q, self.P, price_history)

        return pos, vel, cov


class kf3(trend_detector):
    def __init__(
        self,
        stock_price: List,
        theta,
        H,
        R,
        Q,
        P,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        super().__init__(stock_price, log_returns, name)
        self.theta = theta
        self.H = H
        self.R = R
        self.Q = Q
        self.P = P

    def calc_momentum(self, price_history: List):

        pos, vel, cov = kalman_filter_stock(
            R=self.R,
            Q=self.Q,
            P=self.P,
            price_history=price_history,
            F=self.theta,
            H=self.H,
        )

        return pos, vel, cov


class kf_velocity(trend_detector):
    def __init__(
        self,
        stock_price: List,
        R,
        Q,
        P,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        super().__init__(stock_price, log_returns, name)
        self.R = R
        self.Q = Q
        self.P = P

    def calc_momentum(self, price_history: List):
        n = 10
        if n >= len(price_history):
            n = len(price_history) - 1
        ma = moving_average(stock_price=price_history, n=n, name="ma10")
        _, vel_ma, _ = ma.calc_momentum(price_history=price_history)

        pos, vel, cov = kalman_filter_stock_with_velocity(
            R=self.R,
            Q=self.Q,
            P=self.P,
            price_history=price_history,
            velocity_history=vel_ma,
        )

        return pos, vel, cov


class buy_hold(trend_detector):
    def __init__(
        self, stock_price: List, log_returns: bool = False, name: str = "buy and hold"
    ):

        super().__init__(stock_price, log_returns, name)

    def calc_momentum(self, price_history: List):

        pos = price_history
        vel = [100] * len(price_history)
        cov = [0] * len(price_history)
        return pos, vel, cov


class rsi(trend_detector):
    def __init__(
        self,
        stock_price: List,
        n: int = 14,
        log_returns: bool = False,
        name: str = "buy and hold",
    ):

        super().__init__(stock_price, log_returns, name)
        self.n = n

    def calc_momentum(self, price_history: List):

        rsi_history = RSI(pd.Series(price_history), self.n)
        rsi_history[rsi_history.isna()] = 50
        vel = rsi_history
        vel[(30 < vel) & (vel < 70)] = 50
        vel = -(vel - 50)
        vel[vel < 0] = -1  # Sell
        vel[vel > 0] = 1  # Buy

        pos = price_history
        cov = [0] * len(price_history)

        return pos, list(vel), cov


def hodrick_prescot(price_history: List, lam: float = 1600) -> List:
    price_history_log = price_history
    n_data = len(price_history)

    D = np.zeros([n_data - 2, n_data])
    for t in range(n_data - 2):  # Rows
        for i in range(n_data):  # Columns
            if t == i:
                D[t, i] = 1
            if i == t + 1:
                D[t, i] = -2
            if i == t + 2:
                D[t, i] = 1

    pos_estimate = np.linalg.solve(
        np.eye(n_data) + 2 * lam * np.dot(D.T, D), np.array([price_history_log]).T
    )
    vel_estimate = np.gradient(pos_estimate, edge_order=2, axis=0)
    return list(pos_estimate), list(vel_estimate)


def RSI(close, window_length):
    # Get the difference in price from previous step
    delta = close.diff()

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    return RSI1


class kalman_filter_package(trend_detector):
    def __init__(
        self,
        stock_price: List,
        Q,
        R,
        log_returns: bool = False,
        name: str = "trend_detector",
    ):
        """ """
        super().__init__(stock_price, log_returns, name)
        self.R = R
        self.Q = Q

    def calc_momentum(self, price_history: List):

        kf = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],
            observation_matrices=[1, 0],
            transition_covariance=self.Q,
            observation_covariance=self.R,
        )
        (filtered_state_means, filtered_state_covariances) = kf.filter(price_history)
        pos = list(filtered_state_means[:, 0])
        vel = list(filtered_state_means[:, 1])
        cov = list(filtered_state_covariances)
        return pos, vel, cov
