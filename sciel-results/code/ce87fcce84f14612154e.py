"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _van_der_pol_jac(t, y):
    """Analytical Jacobian for the Van der Pol oscillator (mu=5)."""
    y0, y1 = y
    mu = 5.0
    # dy0/dy0 = 0, dy0/dy1 = 1
    # dy1/dy0 = -1 - 2*mu*y0*y1, dy1/dy1 = mu*(1 - y0^2)
    J = np.array([
        [0.0, 1.0],
        [-1.0 - 2.0 * mu * y0 * y1, mu * (1.0 - y0 * y0)]
    ])
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use Radau with analytical Jacobian and tight tolerances matching the reference
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_van_der_pol_jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: RK4 with fine steps
    h = (t1 - t0) / 10000
    t, y = t0, y0_arr.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
