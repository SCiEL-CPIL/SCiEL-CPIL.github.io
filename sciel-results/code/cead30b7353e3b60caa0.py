"""ODE solver: Robertson problem with adaptive multi-method strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    return np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 3e7 * y1, -1e4 * y1],
        [0.0, 3e7 * y1, 0.0]
    ], dtype=float)


def _enforce_conservation(y, C0):
    """Enforce the conservation law sum(y) = C0."""
    C_end = np.sum(y)
    if C_end > 1e-30 and np.isfinite(C_end):
        return y * (C0 / C_end)
    return y


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem y' = f(t, y) on [t0, t1].
    Returns the final state y(t1).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    C0 = np.sum(y0)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280

    best_y = None
    best_err = np.inf

    # Strategy 1: LSODA with tight tolerances
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='LSODA',
                rtol=1e-14,
                atol=1e-16,
                max_step=(t1 - t0) / 100
            )
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                best_y = y_end
                # Estimate error by comparing with a slightly looser tolerance
                if time.time() < deadline - 20:
                    sol2 = solve_ivp(
                        f,
                        t_span,
                        y0,
                        method='LSODA',
                        rtol=1e-12,
                        atol=1e-14,
                        max_step=(t1 - t0) / 100
                    )
                    if sol2.success and sol2.y.shape[1] > 0:
                        y_end2 = sol2.y[:, -1].copy()
                        y_end2 = _enforce_conservation(y_end2, C0)
                        err = np.linalg.norm(y_end - y_end2) / max(np.linalg.norm(y_end), 1e-30)
                        if err < best_err:
                            best_err = err
    except Exception:
        pass

    # Strategy 2: Radau with analytical Jacobian
    try:
        if time.time() < deadline and (best_y is None or time.time() < deadline - 30):
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-14,
                atol=1e-16,
                jac=_robertson_jac,
                max_step=(t1 - t0) / 50
            )
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                if best_y is not None:
                    err = np.linalg.norm(y_end - best_y) / max(np.linalg.norm(y_end), 1e-30)
                    if err < best_err:
                        best_err = err
                best_y = y_end
    except Exception:
        pass

    # Strategy 3: BDF with analytical Jacobian
    try:
        if time.time() < deadline and (best_y is None or time.time() < deadline - 20):
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-14,
                atol=1e-16,
                jac=_robertson_jac,
                max_step=(t1 - t0) / 50
            )
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                if best_y is not None:
                    err = np.linalg.norm(y_end - best_y) / max(np.linalg.norm(y_end), 1e-30)
                    if err < best_err:
                        best_err = err
                best_y = y_end
    except Exception:
        pass

    # If we still have time and no good result, try RK45 with very small steps
    if time.time() < deadline - 10 and (best_y is None or best_err > 1e-10):
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='RK45',
                rtol=1e-12,
                atol=1e-14,
                max_step=(t1 - t0) / 1000
            )
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1].copy()
                y_end = _enforce_conservation(y_end, C0)
                if best_y is not None:
                    err = np.linalg.norm(y_end - best_y) / max(np.linalg.norm(y_end), 1e-30)
                    if err < best_err:
                        best_err = err
                best_y = y_end
        except Exception:
            pass

    # Final fallback: manual RK4 with conservation enforcement
    if best_y is None:
        h = (t1 - t0) / 5000
        t = t0
        y = y0.copy()
        while t < t1 and time.time() < deadline:
            h_step = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h_step/2, y + h_step/2 * k1)
            k3 = f(t + h_step/2, y + h_step/2 * k2)
            k4 = f(t + h_step, y + h_step * k3)
            y = y + h_step/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h_step
        best_y = _enforce_conservation(y, C0)

    return best_y

# EVOLVE-BLOCK-END
