"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_step(f, t, y, h):
    """One step of the Dormand-Prince RK45. Returns (y5, y_err) where y_err is the
    difference between the 5th-order and 4th-order solutions."""
    # Butcher tableau for Dormand-Prince
    a = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
    B = [
        [],
        [1/5, 0.0],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
    ]
    # 5th order weights
    c5 = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]
    # 4th order weights
    c4 = [5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    k = [None] * 7
    k[0] = f(t, y)
    for i in range(1, 7):
        y_stage = y.copy()
        for j in range(i):
            y_stage = y_stage + h * B[i][j] * k[j]
        k[i] = f(t + a[i] * h, y_stage)

    y5 = y.copy()
    y4 = y.copy()
    for i in range(7):
        y5 = y5 + h * c5[i] * k[i]
        y4 = y4 + h * c4[i] * k[i]
    return y5, y5 - y4


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-11, atol=1e-13):
    """Adaptive RK45 integrator with error-controlled step sizing."""
    t = float(t0)
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 100.0
    SAFETY = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 10.0
    ERR_EXP = 1.0 / 5.0

    # Initial step estimate
    k1 = f(t, y)
    scale = atol + rtol * np.abs(y)
    err_est = np.linalg.norm(k1 / scale)
    if err_est > 1e-10:
        h = h * (1.0 / err_est) ** ERR_EXP
    h = max(h, (t1 - t0) * 1e-6)
    h = min(h, t1 - t0)

    max_steps = 50000
    step_count = 0
    while t < t1 - 1e-14 * abs(t1 - t0) and step_count < max_steps:
        if t + h > t1:
            h = t1 - t
        y_new, y_err = _rk45_step(f, t, y, h)
        step_count += 1
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.linalg.norm(y_err / scale)
        if err_norm <= 1.0:
            t += h
            y = y_new
            if err_norm > 1e-15:
                factor = SAFETY * err_norm ** (-ERR_EXP)
            else:
                factor = MAX_FACTOR
            factor = max(MIN_FACTOR, min(MAX_FACTOR, factor))
            h_new = h * factor
            if t + h_new > t1:
                h_new = t1 - t
            h = h_new
        else:
            factor = SAFETY * err_norm ** (-ERR_EXP)
            factor = max(MIN_FACTOR, factor)
            h = h * factor
            if h < 1e-14 * abs(t1 - t0):
                break
    return y


def integrate(f, t_span, y0):
    """Integrate the Lotka-Volterra IVP using adaptive RK45."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    y = _rk45_adaptive(f, t0, t1, y0, rtol=1e-11, atol=1e-13)
    return y
# EVOLVE-BLOCK-END
