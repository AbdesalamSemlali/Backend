import numpy as np
from sklearn.linear_model import LinearRegression

def cir(yield_curve, dt, num_paths):
    T = len(yield_curve)
    y = np.diff(yield_curve)/np.sqrt(yield_curve[0:-1])
    x1 =  np.sqrt(yield_curve[0:-1])
    x2 = 1/np.sqrt(yield_curve[0:-1])
    X = np.column_stack((x1, x2))
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    a = - model.coef_[0] 
    b = model.coef_[1]/a
    sigma = np.sqrt(np.var(y-(model.coef_[0]*x1 + model.coef_[1]*x2)))
    r0 = yield_curve[0]
    num_steps = int(T / dt)
    times = np.linspace(0, T, num_steps)
    paths = np.zeros((num_paths, num_steps))
    paths[:, 0] = r0

    for i in range(1, num_steps):
        dW = np.random.normal(scale=np.sqrt(dt), size=num_paths)
        paths[:, i] = (
            paths[:, i - 1]
            + a * (b - paths[:, i - 1]) * dt
            + sigma * np.sqrt(paths[:, i - 1]) * dW
        )

    return paths, times , a ,b,sigma