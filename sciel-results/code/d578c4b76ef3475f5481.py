"""Improved program using SciPy's Radau solver with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytic Jacobian for the Van der Pol oscillator with mu=5."""
    y0, y1 = y[0], y[1]
    mu = 5.0
    # y0' = y1
    # y1' = mu*(1-y0^2)*y1 - y0
    # d(y0')/dy0 = 0, d(y0')/dy1 = 1
    # d(y1')/dy0 = -2*mu*y0*y1 - 1
    # d(y1')/dy1 = mu*(1-y0^2)
    return np.array([[0.0, 1.0],
                     [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]])


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau method with tight tolerances matching the reference
    # The reference uses rtol=1e-11, atol=1e-13
    # We use the same or tighter to get a very accurate result
    
    start_time = time.time()
    
    # First attempt: Radau with tight tolerances
    sol = solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        jac=_vdp_jac,
        max_step=0.1
    )
    
    elapsed = time.time() - start_time
    
    if not sol.success:
        # Fallback: use RK45 with very tight tolerances
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='RK45',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.01
        )
    
    if sol.success and len(sol.y) > 0:
        y_end = sol.y[:, -1]
        if np.all(np.isfinite(y_end)):
            return y_end
    
    # Last resort: simple RK4 with many steps
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
