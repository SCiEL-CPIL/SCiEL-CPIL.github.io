import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, y):
    return [10 * (y[1] - y[0]), y[0] * (28 - y[2]) - y[1], y[0] * y[1] - (8.0/3.0) * y[2]]

sol = solve_ivp(lorenz, (0, 2), [1, 1, 1], method='DOP853', rtol=1e-11, atol=1e-13)
print(sol.y[:, -1])
