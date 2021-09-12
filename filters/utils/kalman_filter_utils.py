from typing import List
import numpy as np


class KalmanFilternD:
    def __init__(self, dim_x, dim_z, dim_u=0):
        """
        dim_x = number of state variables to track
        dim_z = number of measurement inputs
        dim_u = size of the control input
        """
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)  # uncertainty covariance
        self.Q = np.eye(dim_x)  # process uncertainty
        self.u = np.zeros((dim_x, 1))  # motion vector
        self.B = 0  # control transition matrix
        self.F = 0  # state transition matrix
        self.H = 0  # measurement function
        self.R = np.eye(dim_z)  # State uncertainty

        self._I = np.eye(dim_x)

        # if use_short_form:
        #    self.update = self.update_short_form

    def update(self, Z, R=None):
        if Z is None:
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # error (residual) between measurement and prediction
        y = Z - np.dot(self.H, self.x)
        # project system uncertainty into measurement space
        S = np.dot(self.H, self.P).dot(self.H.T) + R
        # map system uncertainty into kalman gain
        K = np.dot(self.P, self.H.T).dot(np.linalg.inv(S))
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(K, y)
        I_KH = self._I - np.dot(K, self.H)
        # self.P = np.dot(I_KH,self.P).dot(I_KH.T) + np.dot(K, R).dot(K.T)
        self.P = I_KH.dot(self.P)

    def predict(self, u=0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q


def Q_DWPA(dim, dt=1.0, sigma=1.0):
    """Returns the Q matrix for the Discrete Wiener Process Acceleration Model.
    dim may be either 2 or 3, dt is the time step, and sigma is the variance in
    the noise"""
    assert dim == 2 or dim == 3
    if dim == 2:
        Q = np.array(
            [[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]], dtype=float
        )
    else:
        Q = np.array(
            [
                [0.25 * dt ** 4, 0.5 * dt ** 3, 0.5 * dt ** 2],
                [0.5 * dt ** 3, dt ** 2, dt],
                [0.5 * dt ** 2, dt, 1],
            ],
            dtype=float,
        )
    return Q * sigma


def stock_tracking_filter(
    R, Q, cov=1, F=np.array([[1, 1], [0, 1]]), H=np.array([[1, 0]]), dim_x=2, dim_z=1
):

    """

    :param R: Sensor noise
    :param Q: Process uncertainty
    :param cov: State variable uncertainty
    :return:
    """
    stock_filter = KalmanFilternD(dim_x=dim_x, dim_z=dim_z)
    stock_filter.x = np.array([[0], [0]])
    stock_filter.P *= cov
    stock_filter.Q = Q
    stock_filter.B = 0
    stock_filter.F = F
    stock_filter.H = H
    stock_filter.R *= R
    return stock_filter


def kalman_filter_stock(
    R,
    Q,
    P,
    price_history: List,
    initial_x=None,
    F=np.array([[1, 1], [0, 1]]),
    H=np.array([[1, 0]]),
) -> List:

    price_history_log = price_history
    n_data = len(price_history)

    stock_filter = stock_tracking_filter(R=R, Q=Q, cov=P, F=F, H=H)
    if initial_x is not None:
        stock_filter.x = initial_x
    else:
        stock_filter.x[0, 0] = np.average(price_history_log[0:5])
        stock_filter.x[1, 0] = 0

    count = n_data

    pos = [None] * count
    cov = [None] * count
    vel = [None] * count

    for t in range(count):
        z = price_history_log[t]
        pos[t] = stock_filter.x[0, 0]
        vel[t] = stock_filter.x[1, 0]
        cov[t] = stock_filter.P

        stock_filter.update(z)
        stock_filter.predict()

    return pos, vel, cov


def kalman_filter_stock_with_velocity(
    R,
    Q,
    P,
    price_history: List,
    velocity_history: List,
    initial_x=None,
    F=np.array([[1, 1], [0, 1]]),
    H=np.array([[1, 0], [0, 1]]),
) -> List:
    price_history = list(price_history)
    velocity_history = list(velocity_history)
    n_data = len(price_history)

    stock_filter = stock_tracking_filter(R=R, Q=Q, cov=P, F=F, H=H, dim_z=2)
    if initial_x is not None:
        stock_filter.x = initial_x
    else:
        stock_filter.x[0, 0] = np.average(price_history[0:5])
        stock_filter.x[1, 0] = 0

    count = n_data

    pos = [None] * count
    cov = [None] * count
    vel = [None] * count

    for t in range(count):
        z = np.array([[price_history[t]], [velocity_history[t][0]]])
        pos[t] = stock_filter.x[0, 0]
        vel[t] = stock_filter.x[1, 0]
        cov[t] = stock_filter.P

        stock_filter.update(z)
        stock_filter.predict()

    return pos, vel, cov
