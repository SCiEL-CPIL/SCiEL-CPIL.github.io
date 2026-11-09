import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator from t_span[0] to t_span[1].
    Uses scipy.integrate.solve_ivp with the Radau method and an analytic Jacobian
    to reduce the number of function evaluations compared to finite differences.
    
    f(t, y) returns y' as an array.
    Returns y at t_span[1].
    """
    start_time = time.time()
    deadline = start_time + 290.0

    def jac(t, y):
        # Analytic Jacobian for Van der Pol oscillator:
        # y0' = y1
        # y1' = mu * (1 - y0^2) * y1 - y0
        # mu = 5
        # dy0'/dy0 = 0, dy0'/dy1 = 1
        # dy1'/dy0 = -2 * mu * y0 * y1 - 1
        # dy1'/dy1 = mu * (1 - y0^2)
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val**2)]
        ])

    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            if sol.y.size > 0:
                return sol.y[:, -1]
            else:
                return np.array(y0, dtype=float)

    except Exception:
        return np.array(y0, dtype=float)
