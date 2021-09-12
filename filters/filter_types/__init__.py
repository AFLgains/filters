from filters.filter_types.filter_tune import get_best_params_kf
from filters.filter_types.registry import FilterRegistry
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

R, P, Q = get_best_params_kf()
register = FilterRegistry()
register.register(kalman_filter_package, name = "kf_package", resources={"R": R, "Q": Q})
register.register(kf, name = "kf", resources={"R": R, "P": P, "Q": Q})
register.register(kf_velocity, name = "kf_velocity", resources={"R": R, "P": P, "Q": Q})
register.register(lanczos, name = "lanczos", resources={"n":10})
for d in [10,20,30,40,50]:
    register.register(moving_average, name = "ma%i"%(d), resources={"n":d})
register.register(diff_moving_average, name = "mad", resources={"n1":50,"n2":20})

