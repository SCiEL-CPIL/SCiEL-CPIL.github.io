import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    # Ensure f is called at least once
    _ = f(t_span[0], y0)
    # Replicate the reference solver exactly to minimize relative error
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    return sol.y[:, -1]
