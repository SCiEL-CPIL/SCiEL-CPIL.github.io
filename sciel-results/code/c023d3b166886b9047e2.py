"""Custom adaptive Dormand-Prince (DP5) integrator for the Lotka-Volterra system."""
import numpy as np

# Dormand-Prince (DP5/4) coefficients
# Nodes
_DP_C = np.array([0.0, 2.0/9.0, 1.0/3.0, 4.0/9.0, 5.0/6.0, 1.0, 1.0])

# a_ij (row i uses previous stages)
_DP_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.0/9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/12.0, 3.0/4.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/3.0, 0.0, 4.0/3.0, 0.0, 0.0, 0.0],
    [5.0/12.0, 0.0, 0.0, 8.0/3.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

# b5 (5th order)
_DP_B5 = np.array([1.0/135.0, 0.0, 15.0/1000.0, 75.0/1000.0, 35.0/1000.0, 51031.0/576000.0, 0.0])
# b4 (4th order, for error estimate)
_DP_B4 = np.array([0.0, 0.0, 15.0/1000.0, 75.0/1000.0, 35.0/1000.0, 51031.0/576000.0, 0.0])
# b5 - b4 (error coefficients)
_DP_B_E = _DP_B5 - _DP_B4  # [1/135, 0, 0, 0, 0, 0, -2/9]

# Butcher node times c
_DP_NODES = np.array([0.0, 2.0/9.0, 1.0/3.0, 4.0/9.0, 5.0/6.0, 1.0, 1.0])


def _dp5_step(f, t, y, h):
    """Perform one DP5 step. Returns (y_new, err, nfev)."""
    n = len(y)
    k = np.zeros((7, n))
    k[0] = f(t, y)
    for i in range(1, 7):
        s = y + h * sum(_DP_A[i, j] * k[j] for j in range(i))
        k[i] = f(t + _DP_NODES[i] * h, s)
    y_new = y + h * sum(_DP_B5[j] * k[j] for j in range(7))
    err = h * sum(_DP_B_E[j] * k[j] for j in range(7))
    return y_new, err, 7


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0

    rtol = 1e-13
    atol = 1e-15
    min_h = 1e-10
    max_h = 1e-3
    # Start with a small step
    h = 1e-4

    # Safety and exponents for step size control
    SAFETY = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 5.0
    EXPO = 1.0 / 5.0  # order 5

    n = len(y)
    nfev = 0
    total_span = t1 - t0

    while t < t1:
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h

        y_new, err, nfev_step = _dp5_step(f, t, y, h)
        nfev += nfev_step

        # Error norm: scaled by rtol*|y| + atol
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.linalg.norm(err / scale) / np.sqrt(n)

        if not np.all(np.isfinite(y_new)):
            # Step failed, reduce h drastically
            h *= 0.1
            if h < min_h * 0.1:
                break
            continue

        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new

            # Choose new step size
            if err_norm < 1e-12:
                h_new = h * MAX_FACTOR
            else:
                h_new = SAFETY * h * (err_norm ** (-EXPO))
            h = min(max(h_new, h * MIN_FACTOR), h * MAX_FACTOR)
            h = min(h, max_h)
            h = max(h, min_h)
        else:
            # Reject step, reduce h
            if err_norm > 0:
                h_new = SAFETY * h * (err_norm ** (-EXPO))
            else:
                h_new = h * 0.5
            h = max(h_new, min_h)

    return y
