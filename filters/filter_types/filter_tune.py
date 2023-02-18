from bayes_opt import BayesianOptimization
from filters.filter_types.filters import kf, kf3
import numpy as np
import math

NAN = float("nan")


def find_optimal_kalman_filter(price_history_list):
    def objective(R, qxx, qyy, qxy):
        k_filter = kf(
            stock_price=price_history_list,
            R=R,
            P=10,
            Q=np.array([[np.exp(qxx), qxy], [qxy, np.exp(qyy)]]),
            name="kf1",
        )
        res, ex_ante_pos, ex_ante_vel = k_filter.back_test(
            initial_capital=100, verbose=False
        )
        return res["win_rate"]

    pbounds = {"R": (0.1, 20), "qxx": (-8, -2), "qyy": (-8, -2), "qxy": (-0.1, 0.1)}

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={
            "R": 8.332254532919853,
            "qxx": -3.8341625098938406,
            "qxy": -0.040633053397636856,
            "qyy": -6.234843359486865,
        },
        lazy=True,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

    return optimizer.max


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def find_optimal_kalman_filter_k3(price_history_list, velocity_list):
    def objective(R, qxx, qyy, qxy, x1, x2, x3, h1, h2):
        k_filter = kf3(
            stock_price=price_history_list,
            R=R,
            P=10,
            Q=np.array([[np.exp(qxx), qxy], [qxy, np.exp(qyy)]]),
            name="kf3",
            theta=np.array([[x1, x2], [0, x3]]),
            H=np.array([[h1, h2]]),
        )

        res, ex_ante_pos, ex_ante_vel = k_filter.back_test(
            initial_capital=100, verbose=False
        )
        neg_rsme_pos = (
            -1
            / (np.mean(np.array(price_history_list)))
            * rmse(np.array(ex_ante_pos), np.array(price_history_list))
        )
        neg_rsme_vel = (
            -1
            / (np.mean(np.array(velocity_list)))
            * rmse(np.array(ex_ante_pos), np.array(velocity_list))
        )

        return neg_rsme_pos + neg_rsme_vel

    pbounds = {
        "R": (0.1, 20),
        "qxx": (-8, -2),
        "qyy": (-8, -2),
        "qxy": (-0.1, 0.1),
        "x1": (0.1, 2),
        "x2": (0.1, 2),
        "x3": (0.1, 2),
        "h1": (0.1, 2),
        "h2": (0, 2),
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={
            "R": 8.332254532919853,
            "qxx": -3.8341625098938406,
            "qxy": -0.040633053397636856,
            "qyy": -6.234843359486865,
            "x1": 1,
            "x2": 1,
            "x3": 1,
            "h1": 1,
            "h2": 0,
        },
        lazy=True,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=500,
    )

    print(optimizer.max)

    return optimizer.max
