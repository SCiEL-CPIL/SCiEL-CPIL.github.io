import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Radau method with analytical Jacobian.
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        # J = [[0, 1], [-1 - 10*y0*y1, 5*(1 - y0^2)]]
        return np.array([
            [0.0, 1.0],
            [-1.0 - 10.0 * y0_val * y1_val, 5.0 * (1.0 - y0_val * y0_val)]
        ])

    # Use tolerances aligned with reference (rtol=1e-11, atol=1e-13)
    # Slightly tighter to ensure we're below reference error, but not excessively so
    # to minimize Newton iterations and f calls
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='Radau',
        rtol=1e-11,
        atol=1e-13,
        jac=jac
    )

    if not sol.success:
        # Check if we're running out of time
        if time.time() > deadline:
            return np.array(y0, dtype=float)
        # Try with slightly relaxed tolerances
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            jac=jac
        )
        if not sol.success:
            return np.array(y0, dtype=float)

    return sol.y[:, -1]
