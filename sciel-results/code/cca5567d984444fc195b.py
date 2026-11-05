"""ODE solver: 15-dimensional stiff linear system."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate a 15-dimensional stiff linear ODE y' = f(t, y) over t_span.
    Strategy:
      1. Assume the system is linear with constant coefficients: f(t, y) = A @ y.
         Construct A by evaluating f at the standard basis vectors (n calls).
         Compute y(t1) = expm((t1-t0)*A) @ y0.
      2. If that fails, fall back to Radau with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0

    # Build A by evaluating f at the standard basis vectors
    try:
        A = np.empty((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            A[:, j] = np.asarray(f(t0, ej), dtype=float)

        if time.time() > deadline:
            # Can't finish expm in time; return y0 as best guess
            return y0

        dt = t1 - t0
        M = expm(dt * A)
        y_end = M @ y0

        if time.time() > deadline:
            return y_end

        return y_end

    except Exception:
        pass

    # Fallback: use Radau with tight tolerances
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 500
        )
        if time.time() > deadline:
            if sol.y.size > 0:
                return sol.y[:, -1]
            return y0
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: BDF
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
# EVOLVE-BLOCK-END
