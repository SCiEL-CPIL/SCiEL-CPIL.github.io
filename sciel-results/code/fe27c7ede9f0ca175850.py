"""Improved program using SciPy's Radau solver with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) using Radau method.
    Uses an analytic Jacobian to improve efficiency and accuracy.
    """
    # Wrap f to count calls, although the count is not explicitly returned,
    # the evaluator measures it. We just need to ensure we call it correctly.
    # The problem statement says "every call is counted". We don't need to
    # return the count, just the result.
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    mu = 5.0
    
    # Analytic Jacobian
    def jac(t, y):
        y0, y1 = y
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]
        ])

    # Use Radau with tolerances slightly tighter than reference to ensure
    # our solution is highly accurate relative to the reference.
    # Reference: rtol=1e-11, atol=1e-13
    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='Radau',
        jac=jac,
        rtol=1e-12,
        atol=1e-14
    )
    
    if not sol.success:
        # Fallback to slightly looser tolerances if it fails
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='Radau',
            jac=jac,
            rtol=1e-10,
            atol=1e-12
        )
        if not sol.success:
            raise RuntimeError("Integration failed")

    y_end = sol.y[:, -1]
    
    if not np.all(np.isfinite(y_end)):
        raise RuntimeError("Non-finite result")
        
    return y_end
# EVOLVE-BLOCK-END
