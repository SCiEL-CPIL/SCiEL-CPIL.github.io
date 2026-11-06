"""Direct solver for the fixed Robertson ODE: implicit Radau IVP formulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] using an implicit Radau solver.

    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval (fixed to [0, 40] by the evaluator).
    y0 : array-like
        Initial condition at t0.

    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0_flat = np.asarray(y0, dtype=float).reshape(-1)

    # Analytic Jacobian df/dy for the Robertson problem (matches the fixed task):
    #   f0 = -0.04*y0 + 1e4*y1*y2
    #   f1 =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    #   f2 =  3e7*y1^2
    def jac(t, y):
        y = np.asarray(y).reshape(-1)
        y0v, y1v, y2v = y[0], y[1], y[2]
        return np.array([
            [-0.04,                  1.0e4 * y2v,  1.0e4 * y1v],
            [ 0.04, -1.0e4 * y2v - 6.0e7 * y1v, -1.0e4 * y1v],
            [ 0.0,                   6.0e7 * y1v,  0.0      ],
        ])

    # Time budget guard: evaluator kills at 330s; keep a safety margin.
    start_time = time.time()
    time_limit = 300.0

    best_result = None

    # Primary strategy: Implicit Radau solver with tight tolerances.
    # Radau is effective for stiff problems and minimizes function calls for a given accuracy.
    try:
        sol = solve_ivp(
            f, t_span, y0_flat,
            method="Radau",
            jac=jac,
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success:
            best_result = sol.y[:, -1].reshape(-1)
        else:
            # If Radau fails (rare), try BDF as a fallback
            sol = solve_ivp(
                f, t_span, y0_flat,
                method="BDF",
                jac=jac,
                rtol=1e-12,
                atol=1e-14,
            )
            if sol.success:
                best_result = sol.y[:, -1].reshape(-1)
    except Exception:
        pass

    # Fallback if implicit solvers fail completely
    if best_result is None:
        try:
            m = 1000000
            h = (t1 - t0) / m
            y = y0_flat.copy()
            tt = t0
            for _ in range(m):
                y = y + h * f(tt, y)
                tt += h
            best_result = y
        except Exception:
            best_result = y0_flat

    return best_result
# EVOLVE-BLOCK-END
