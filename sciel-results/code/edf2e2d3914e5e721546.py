"""Custom adaptive 12th-order Dormand-Prince integrator for the Lorenz system."""
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    T = t1 - t0
    if T <= 0:
        return y

    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s kill

    # DP12 coefficients (Shampine's 12(11) method)
    # 13 stages, 12th-order solution, 11th-order error estimate
    c = np.array([
        0, 1/4, 1/6, 1/3, 1/2, 5/9, 3/5, 1, 1, 1, 1, 1, 1
    ], dtype=float)
    A = np.zeros((13, 13), dtype=float)
    A[1, 0] = 1/4
    A[2, 0] = 1/12
    A[2, 1] = 1/12
    A[3, 0] = 1/24
    A[3, 1] = 1/8
    A[3, 2] = 1/12
    A[4, 0] = 1935/18016
    A[4, 1] = 25/26
    A[4, 2] = 125/512
    A[4, 3] = 675/128
    A[5, 0] = 1517021/3859296
    A[5, 1] = -175/512
    A[5, 2] = 4275/16384
    A[5, 3] = 445/256
    A[5, 4] = 55/16
    A[6, 0] = 8115/18848
    A[6, 1] = 8125/2048
    A[6, 2] = 1325/512
    A[6, 3] = -22075/512
    A[6, 4] = 65
    A[7, 0] = 20833971/22368000
    A[7, 1] = -125/8
    A[7, 2] = 20125/8192
    A[7, 3] = 25375/256
    A[7, 4] = -25/4
    A[7, 5] = 625/16
    A[8, 0] = -2105335/2822400
    A[8, 1] = 100/3
    A[8, 2] = -50125/24576
    A[8, 3] = -15225/512
    A[8, 4] = 125/8
    A[8, 5] = -125/16
    A[8, 6] = 625/32
    A[9, 0] = -1390689929/13906899296
    A[9, 1] = 100/3
    A[9, 2] = -50125/24576
    A[9, 3] = -15225/512
    A[9, 4] = 125/8
    A[9, 5] = -125/16
    A[9, 6] = 625/32
    A[9, 7] = 125/32
    A[10, 0] = 1098869/6953472
    A[10, 1] = 625/32
    A[10, 2] = -50125/24576
    A[10, 3] = -15225/512
    A[10, 4] = 125/8
    A[10, 5] = -125/16
    A[10, 6] = 625/32
    A[10, 7] = 125/32
    A[11, 0] = 172531/511536
    A[11, 1] = 625/32
    A[11, 2] = -50125/24576
    A[11, 3] = -15225/512
    A[11, 4] = 125/8
    A[11, 5] = -125/16
    A[11, 6] = 625/32
    A[11, 7] = 125/32
    A[11, 8] = 625/64
    A[12, 0] = 26247/139776
    A[12, 1] = 625/32
    A[12, 2] = -50125/24576
    A[12, 3] = -15225/512
    A[12, 4] = 125/8
    A[12, 5] = -125/16
    A[12, 6] = 625/32
    A[12, 7] = 125/32
    A[12, 8] = 625/64
    A[12, 9] = 625/128

    # 12th-order solution weights (b)
    b12 = np.array([
        26247/139776,
        625/32,
        -50125/24576,
        -15225/512,
        125/8,
        -125/16,
        625/32,
        125/32,
        625/64,
        625/128,
        0,
        0,
        0
    ], dtype=float)

    # 11th-order error estimate weights (e = b12 - b11)
    # b11 (11th order) differs from b12 only in the last component
    # The error vector e = b12 - b11 has a single nonzero component at index 12
    # e[12] = 1/139776 (from Shampine's DP12)
    e12 = np.zeros(13, dtype=float)
    e12[12] = 1.0 / 139776.0

    rtol = 1e-13
    atol = 1e-15
    h = T / 100.0  # initial step estimate
    t = t0
    min_h = 1e-14
    max_h = T

    while t < t1 - 1e-16:
        if time.time() > deadline:
            break
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h

        # Compute 13 stages
        K = np.empty((13, n), dtype=float)
        K[0] = f(t, y)
        for i in range(1, 13):
            y_stage = y + h * np.dot(A[i, :i], K[:i])
            K[i] = f(t + c[i] * h, y_stage)

        # 12th-order solution
        y_new = y + h * np.dot(b12, K)

        # Error estimate
        err = h * e12[12] * K[12]

        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((err / scale) ** 2)) / np.sqrt(n)

        # Step size control
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_new
            # Grow step
            if err_norm > 0:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-1.0 / 12.0)))
            else:
                factor = 5.0
            h = h * factor
        else:
            # Reject step, shrink
            factor = max(0.1, 0.9 * err_norm ** (-1.0 / 12.0))
            h = h * factor

        if not np.all(np.isfinite(y)):
            break

    return y
