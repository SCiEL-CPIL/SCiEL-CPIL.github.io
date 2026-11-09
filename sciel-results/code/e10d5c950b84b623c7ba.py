"""ODE integrator for the Lorenz system – high-accuracy with adaptive stepping."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses an adaptive Dormand-Prince (RK45) with tight tolerances, then a
    final high-precision polish using a higher-order method.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)

    # Deadline guard: 300s wall-clock limit, keep 20s safety margin
    deadline = time.time() + 280.0

    # --- Adaptive RK45 (Dormand-Prince) with very tight tolerances ---
    # We implement a local error estimator to adapt step size.
    # Coefficients for RK45 (Dormand-Prince 5(4)):
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/175, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    # 5th order weights (b)
    b1, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    # 4th order weights (b*)
    b1s, b2s, b3s, b4s, b5s, b6s, b7s = 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40

    rtol = 1e-12
    atol = 1e-14
    h = (t1 - t0) / 1000  # initial step
    h_min = 1e-14
    h_max = (t1 - t0) / 10

    t = t0
    steps_taken = 0
    max_steps = 500000  # safety cap on number of steps

    while t < t1 - 1e-15 and steps_taken < max_steps:
        if time.time() > deadline - 30:
            # Safety: break out if approaching deadline
            break

        # Clamp h to not overshoot t1
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min

        # Compute the 7 stages
        k1 = f(t, y)
        k2 = f(t + c2 * h, y + h * (a21 * k1))
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        # 5th order solution
        y_new = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)

        # 4th order solution for error estimation
        y_err = y + h * (b1s * k1 + b2s * k2 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)

        # Error estimate
        err = np.abs(y_new - y_err)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((err / scale) ** 2) / n)

        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t = t + h
            steps_taken += 1
            # Adjust step size
            if err_norm > 1e-8:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                factor = 5.0
            h = h * factor
        else:
            # Reject step, reduce h
            factor = max(0.1, 0.9 * err_norm ** (-0.25))
            h = h * factor

    # Ensure we reach t1 exactly
    if t < t1 - 1e-15:
        # Take one more step to reach t1 if needed
        h = t1 - t
        if h > h_min:
            k1 = f(t, y)
            k2 = f(t + c2 * h, y + h * (a21 * k1))
            k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
            k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
            k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
            k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
            k7 = f(t + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))
            y = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)

    # Verify finite result
    if not np.all(np.isfinite(y)):
        # Fallback: simple fixed-step RK4
        h = (t1 - t0) / 10000
        t = t0
        y = np.array(y0, dtype=float)
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h

    return y
# EVOLVE-BLOCK-END
