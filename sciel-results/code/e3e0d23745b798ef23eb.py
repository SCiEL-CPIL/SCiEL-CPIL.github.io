"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Deadline guard
START_TIME = time.time()
DEADLINE = 320.0  # 10s margin before 330s kill

def _fallback_rk4(f, t_span, y0):
    """Fallback RK4 if SciPy is not available or fails."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > DEADLINE:
            return y
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    if not HAS_SCIPY:
        return _fallback_rk4(f, t_span, y0)

    # Use DOP853 with tight tolerances to minimize rel error.
    # Reference used rtol=1e-11, atol=1e-13.
    # We use slightly tighter to ensure rel is at the floor (1e-14) if possible.
    # DOP853 is efficient, so call count should remain low.
    
    # We wrap f to count calls if we wanted to, but solve_ivp doesn't expose that directly easily.
    # The task says "every call is counted" by the evaluator, so we don't need to count them ourselves.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.1 # Optional: prevent huge steps if needed, but DOP853 handles adaptivity well
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if solve_ivp fails
            return _fallback_rk4(f, t_span, y0)
    except Exception:
        return _fallback_rk4(f, t_span, y0)

# EVOLVE-BLOCK-END
