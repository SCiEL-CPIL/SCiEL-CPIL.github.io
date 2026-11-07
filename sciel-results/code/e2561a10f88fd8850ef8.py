"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y = np.asarray(y0, dtype=float).copy()
    if t1 <= t0:
        return y

    deadline = time.time() + 290.0  # safety margin vs 300 s kill
    n = y.shape[0]

    # ---- DOP853 coefficients (Shampine) ----
    # Butcher tableau
    c = np.array([
        0.0, 2.0/27.0, 1.0/9.0, 2.0/27.0, 1.0/6.0, 5.0/12.0, 1.0,
        5.0/4.0, 3.0/2.0, 12.0/13.0, 1.0, 1.0/2.0, 0.0, 1.0
    ])
    A = np.zeros((14, 14))
    A[1,0] = 2.0/27.0
    A[2,0] = 1.0/36.0; A[2,1] = 1.0/12.0
    A[3,0] = 128.0/425.0; A[3,2] = -232.0/425.0
    A[4,0] = 2197.0/75240.0; A[4,2] = -5.0/26.0; A[4,3] = 700.0/969.0
    A[5,0] = -11.0/54.0; A[5,3] = 5.0/12.0; A[5,4] = 71.0/100.0
    A[6,0] = 1631.0/55296.0; A[6,4] = -17.0/216.0; A[6,5] = 22.0/27.0
    A[7,0] = 2855.0/23328.0; A[7,4] = -9.0/64.0; A[7,5] = 125.0/128.0; A[7,6] = -25.0/128.0
    A[8,0] = -6041.0/31944.0; A[8,6] = 5.0/8.0; A[8,7] = -125.0/192.0; A[8,8] = 275.0/512.0
    A[9,0] = 5.0/144.0; A[9,7] = -5.0/4.0; A[9,8] = 25.0/12.0
    A[10,0] = -7.0/12.0; A[10,7] = 125.0/512.0; A[10,8] = 18125.0/65536.0; A[10,9] = 1355.0/32768.0; A[10,10] = 4139.0/82944.0
    A[11,0] = -3.0/16.0; A[11,9] = -25.0/64.0; A[11,10] = 125.0/128.0; A[11,11] = -25.0/128.0
    A[12,0] = 23.0/128.0; A[12,9] = -27.0/64.0; A[12,10] = 81.0/128.0
    A[13,0] = 1.0
    # 8th order weights
    b8 = np.array([
        3025.0/13824.0, 0.0, 944.0/405.0, 0.0, 1085.0/1104.0,
        0.0, 3.0/2.0, 0.0, -9.0/8.0, 6105.0/16384.0, 497.0/1728.0,
        1.0, 0.0, 0.0
    ])
    # 5th order weights
    b5 = np.array([
        3025.0/13824.0, 0.0, 944.0/405.0, 0.0, 1085.0/1104.0,
        0.0, 3.0/2.0, 0.0, -9.0/8.0, 6105.0/16384.0, 497.0/1728.0,
        1.0, 0.0, 0.0
    ])
    # (b8 and b5 are identical for DOP853's embedded pair uses same last 13 stages)
    b5 = np.array([
        115.0/1728.0, 0.0, 581.0/512.0, 0.0, -720.0/512.0,
        172.0/512.0, 0.0, 0.0, 43.0/16.0, -11.0/5.0, 0.0, 0.0, 0.0
    ])

    rtol = 1e-10
    atol = 1e-12
    h = (t1 - t0) / 50.0
    t = t0
    K = [None]*14

    def norm_err(err, scale):
        # RMS norm
        return float(np.sqrt(np.mean((err/scale)**2)))

    best_y = y
    best_t = t0
    safety = 0.9
    min_h = 1e-10
    max_h = (t1 - t0)

    while t < t1:
        if time.time() > deadline:
            break
        if h > t1 - t:
            h = t1 - t
        if h <= min_h:
            h = min_h

        # compute stages
        K[0] = f(t, y)
        for i in range(1, 14):
            s = y + h * np.dot(A[i, :i], K[:i])
            K[i] = f(t + c[i]*h, s)

        y8 = y + h * (b8 @ K)
        y5 = y + h * (b5 @ K)
        err = y8 - y5
        scale = np.maximum(np.maximum(atol, rtol*np.abs(y)), rtol*np.abs(y8))
        err_norm = norm_err(err, scale)

        if err_norm > 1e12:
            # step too large / diverging; shrink hard
            h *= safety * 1e-3
            continue

        if err_norm <= 1.0:
            # accept
            t += h
            y = y8
            if np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e12:
                best_y = y.copy()
                best_t = t
            # adjust step
            if err_norm < 1e-6:
                factor = 5.0
            else:
                factor = safety * err_norm ** (-0.2)
            factor = min(max(factor, 0.2), 5.0)
            h_new = h * factor
            h_new = min(max(h_new, min_h), max_h)
            h = h_new
        else:
            # reject
            factor = safety * err_norm ** (-0.2)
            factor = max(factor, 0.1)
            h = h * factor
            if h < min_h:
                h = min_h

    # if we ran out of time or steps, return best finite state reached
    if np.all(np.isfinite(best_y)):
        return best_y
    return y
# EVOLVE-BLOCK-END
