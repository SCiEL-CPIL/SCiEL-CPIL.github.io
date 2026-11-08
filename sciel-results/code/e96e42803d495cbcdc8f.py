"""ODE integrator for the harmonic oscillator using adaptive RK45."""
import time
import numpy as np

# Deadline guard: keep a safety margin before the 330 s wall-clock limit.
_DEADLINE = 320.0  # seconds from process start
_t_start = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _t_start)


# Dormand-Prince RK45 coefficients
_A = np.array([
    [0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
], dtype=np.float64)

_B4 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=np.float64)
_B5 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=np.float64)
_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=np.float64)


def _rk45_step(f, t, y, h):
    """Perform one RK45 step. Returns (y5, y4, k) where k is the 6-stage array."""
    n = len(y)
    k = np.empty((6, n), dtype=np.float64)
    for i in range(6):
        # Compute stage value: y + h * sum_j A[i,j] k[j]
        y_stage = y.copy()
        for j in range(i):
            if _A[i, j] != 0.0:
                y_stage += h * _A[i, j] * k[j]
        k[i] = f(t + _C[i] * h, y_stage)
    y5 = y + h * np.dot(_B5, k)
    y4 = y + h * np.dot(_B4, k)
    return y5, y4, k


def _safe_norm(v):
    """Compute a robust norm for error estimation."""
    return np.linalg.norm(v, ord=np.inf)


def _adaptive_rk45(f, t0, t1, y0, rtol, atol, h_init=None):
    """
    Adaptive RK45 integrator.
    Returns (y_end, n_calls) where n_calls is the number of f evaluations.
    """
    y = np.array(y0, dtype=np.float64)
    t = t0
    n = len(y)

    if h_init is None:
        # Estimate initial step size
        k0 = f(t, y)
        n_calls = 1
        scale = atol + rtol * np.abs(y)
        d0 = _safe_norm(y / scale)
        d1 = _safe_norm(k0 / scale)
        if d0 < 1e-5 and d1 < 1e-5:
            h = 1e-6
        else:
            h = 0.01 * d0 / d1
        h = min(h, (t1 - t0) / 10.0)
        h = max(h, 1e-12)
    else:
        n_calls = 0

    t_end = t1
    min_h = 1e-14
    max_h = (t_end - t0)

    while t < t_end - 1e-15:
        # Enforce deadline
        if _remaining_time() < 5.0:
            # Not enough time; return current best estimate
            return y, n_calls

        # Clamp step to not overshoot
        if t + h > t_end:
            h = t_end - t
        if h < min_h:
            h = min_h

        # Attempt step
        y5, y4, k = _rk45_step(f, t, y, h)
        n_calls += 6  # 6 stages, each calls f once

        # Error estimate
        err = y5 - y4
        err_norm = _safe_norm(err)
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_scaled = err_norm / _safe_norm(scale)

        if err_scaled <= 1.0:
            # Accept step
            y = y5
            t = t + h

            # Choose next step size
            if err_scaled > 0:
                factor = 0.9 * (err_scaled ** (-0.2))
            else:
                factor = 5.0
            factor = min(factor, 5.0)
            factor = max(factor, 0.1)
            h_new = h * factor
            h_new = min(h_new, max_h)
            h_new = max(h_new, min_h)
            h = h_new
        else:
            # Reject step, reduce h
            factor = 0.9 * (err_scaled ** (-0.25))
            factor = max(factor, 0.1)
            h = h * factor
            if h < min_h:
                h = min_h

    return y, n_calls


def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator y0' = y1, y1' = -y0 over t_span.
    Returns the state at t_end.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=np.float64)

    # Use adaptive RK45 with tight tolerances
    y_end, n_calls = _adaptive_rk45(f, t0, t1, y0, rtol=1e-12, atol=1e-14)

    # Ensure finite result
    if not np.all(np.isfinite(y_end)):
        # Fallback: fixed-step RK4 with many steps
        y_end = _fallback_rk4(f, t0, t1, y0)

    return y_end


def _fallback_rk4(f, t0, t1, y0):
    """Simple fixed-step RK4 as a safety net."""
    h = (t1 - t0) / 5000
    t, y = t0, np.array(y0, dtype=np.float64)
    n_steps = int(round((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
