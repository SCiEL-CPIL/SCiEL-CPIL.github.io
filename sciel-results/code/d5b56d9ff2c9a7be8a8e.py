"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using Richardson extrapolation
    with DOP853 at two different max_step values to achieve effective higher-order
    accuracy. Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Very tight tolerances for high accuracy
    rtol = 1e-13
    atol = 1e-15
    
    # Call counter to track evaluations
    call_count = 0
    original_f = f
    
    def counting_f(t, y):
        nonlocal call_count
        call_count += 1
        return original_f(t, y)
    
    # Base max_step for first DOP853 run
    base_max_step = 0.01
    # Halved max_step for second DOP853 run
    half_max_step = base_max_step / 2.0
    
    # First DOP853 solve with base max_step
    sol1 = solve_ivp(
        counting_f,
        [t0, t1],
        y0,
        method='DOP853',
        rtol=rtol,
        atol=atol,
        max_step=base_max_step
    )
    
    if not sol1.success:
        # Fallback: use RK4 with many steps
        n_steps = 10000
        h = (t1 - t0) / n_steps
        t, y = t0, y0.copy()
        for _ in range(n_steps):
            k1 = counting_f(t, y)
            k2 = counting_f(t + h / 2, y + h / 2 * k1)
            k3 = counting_f(t + h / 2, y + h / 2 * k2)
            k4 = counting_f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    # Second DOP853 solve with halved max_step
    sol2 = solve_ivp(
        counting_f,
        [t0, t1],
        y0,
        method='DOP853',
        rtol=rtol,
        atol=atol,
        max_step=half_max_step
    )
    
    if not sol2.success:
        # If second solve fails, return the first one
        return sol1.y[:, -1]
    
    # Richardson extrapolation for DOP853 (order 8)
    # y_ext = (2^8 * y2 - y1) / (2^8 - 1)
    order = 8
    factor = 2.0 ** order
    y1 = sol1.y[:, -1]
    y2 = sol2.y[:, -1]
    y_extrapolated = (factor * y2 - y1) / (factor - 1.0)
    
    return y_extrapolated
# EVOLVE-BLOCK-END
