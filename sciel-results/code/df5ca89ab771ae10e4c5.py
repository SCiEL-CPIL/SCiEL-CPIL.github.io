"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system from t_span[0] to t_span[1]
    with initial condition y0, returning the state at t_span[1].
    
    Uses scipy.integrate.solve_ivp with DOP853 and very tight tolerances,
    no max_step constraint, to minimize function calls while achieving
    relative error at the 1e-14 floor.
    """
    from scipy.integrate import solve_ivp
    
    t_start = time.time()
    deadline = 325.0  # seconds, safety margin before 330s kill
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # DOP853 with very tight tolerances, no max_step restriction.
    # The Lotka-Volterra system is smooth and non-stiff over [0,5],
    # so DOP853 can take large steps efficiently.
    # Tighter tolerances than reference to push rel error to the 1e-14 floor.
    result = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
    )
    
    if not result.success or result.y.shape[1] == 0:
        # Fallback: retry with reference tolerances
        if time.time() - t_start < deadline - 5:
            result = solve_ivp(
                f,
                (t0, t1),
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13,
            )
    
    if result.success and result.y.shape[1] > 0:
        y_end = result.y[:, -1]
    else:
        # Last resort: fixed-step RK4
        y_end = _rk4_fallback(f, t0, t1, y0, 10000)
    
    # Ensure the result is finite
    if not np.all(np.isfinite(y_end)):
        y_end = _rk4_fallback(f, t0, t1, y0, 10000)
    
    return np.asarray(y_end, dtype=float)


def _rk4_fallback(f, t0, t1, y0, steps):
    """Simple RK4 fallback in case scipy solver fails."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
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
