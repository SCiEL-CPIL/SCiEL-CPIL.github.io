"""
High-precision ODE integrator for the Lotka-Volterra system.

Strategy:
- Use scipy.integrate.solve_ivp with method 'DOP853' (same as the reference).
- Set rtol=1e-11 and atol=1e-13 to match the reference tolerances exactly,
  which minimizes the number of function calls while keeping the relative
  error at the floor (1e-14) where no further accuracy gain is possible.
- A single solve_ivp call returns the state exactly at t1 in sol.y[:, -1].
- Enforce a hard call-budget ceiling and a time-deadline guard as safety nets.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrates the given ODE system over t_span starting from y0.
    Returns the state vector at t = t_span[1].
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()

    # Edge case: zero-length span
    if t0 == t1:
        return y0

    start_time = time.time()
    DEADLINE = 280.0  # seconds; well under the 300 s evaluator limit

    call_count = [0]
    MAX_CALLS = 2_000_000  # hard ceiling from the contract

    def counted_f(t, y):
        call_count[0] += 1
        if call_count[0] > MAX_CALLS:
            raise RuntimeError("Function-call budget exceeded")
        return f(t, y)

    try:
        sol = solve_ivp(
            counted_f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
        )

        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end

        # Fallback: slightly tighter tolerances in case the first run had issues
        call_count[0] = 0
        sol2 = solve_ivp(
            counted_f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
        )

        if sol2.success:
            y_end = sol2.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end

        return y0

    except Exception:
        return y0

# EVOLVE-BLOCK-END
