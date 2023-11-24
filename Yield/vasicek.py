import numpy as np
from scipy.stats import linregress


def vasicek_model(yield_curve, dt, num_paths):
    T = len(yield_curve)
    y = np.diff(yield_curve)
    x =  yield_curve[0:-1]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    a = - slope 
    b = intercept/a
    sigma = np.sqrt(np.var(y-(intercept +slope*x)))
    r0 = yield_curve[0]
    num_steps = int(T / dt)
    times = np.linspace(0, T, num_steps )
    paths = np.zeros((num_paths, num_steps))
    paths[:, 0] = r0

    for i in range(1, num_steps):
        dW = np.random.normal(scale=np.sqrt(dt), size=num_paths)
        paths[:, i] = (
            paths[:, i - 1]
            + a * (b - paths[:, i - 1]) * dt
            + sigma * dW
        )

    return paths , times , a , b , sigma