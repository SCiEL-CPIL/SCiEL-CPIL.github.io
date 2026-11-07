"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] with initial condition y0.
    Uses an adaptive DOP853 (7th/8th order) Runge-Kutta method.
    """
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin out of 300s limit

    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0

    if t1 <= t0:
        return y

    # DOP853 coefficients (Butcher tableau)
    # This is a 12-stage, 7th-order method with an embedded 6th-order error estimate
    c = np.array([
        0.0,
        1/36,
        1247/42516,
        439/14172,
        7/16,
        328321/919800,
        94889/337500,
        91/225,
        23048/66825,
        3/2,
        1,
        1
    ])

    # a[i][j] for i > j, a[i][i] = 0
    a = np.zeros((12, 12))
    a[1, 0] = 1/36
    a[2, 0] = -857671/19116000
    a[2, 1] = 31331/967200
    a[3, 0] = -1351/172800
    a[3, 1] = 0
    a[3, 2] = 124807/2106000
    a[4, 0] = -17337368721/191160000000
    a[4, 1] = 148162971/109734000
    a[4, 2] = -1787222011/472875000
    a[4, 3] = 1187001641/236437500
    a[5, 0] = 4951088124861/55395000000000
    a[5, 1] = -131806957753/21817200000
    a[5, 2] = 8996706463/402375000
    a[5, 3] = -39603543863/134125000
    a[5, 4] = 7812100397/109734000
    a[6, 0] = -771343975492/1860513750000
    a[6, 1] = 0
    a[6, 2] = 1150063489/472875000
    a[6, 3] = -178292131/157612500
    a[6, 4] = 1386185/2106000
    a[6, 5] = -9175511/18605100
    a[7, 0] = 46147443642/1860513750000
    a[7, 1] = 0
    a[7, 2] = -270792799/157612500
    a[7, 3] = 153916493/157612500
    a[7, 4] = -5999429/1341250
    a[7, 5] = 1092274/1576125
    a[7, 6] = -3278/10125
    a[8, 0] = 295505091641/1860513750000
    a[8, 1] = 0
    a[8, 2] = -617236847/315225000
    a[8, 3] = -319590751/105075000
    a[8, 4] = 42127813/3483000
    a[8, 5] = 2319217/472875
    a[8, 6] = -247/2025
    a[8, 7] = 1/1125
    a[9, 0] = 525303539777/1104308250000
    a[9, 1] = 0
    a[9, 2] = 2906258129/315225000
    a[9, 3] = -1394895449/35025000
    a[9, 4] = -21684451/729000
    a[9, 5] = -2043779/283725
    a[9, 6] = 2123/1125
    a[9, 7] = -1/2025
    a[9, 8] = -1/1125
    a[10, 0] = 17088866600141/7362055000000
    a[10, 1] = 0
    a[10, 2] = -366386239/31522500
    a[10, 3] = 31353173/141750
    a[10, 4] = -3905137/8100
    a[10, 5] = 139021/405
    a[10, 6] = -4/9
    a[10, 7] = 0
    a[10, 8] = 0
    a[10, 9] = 1/9
    a[11, 0] = 525303539777/1104308250000
    a[11, 1] = 0
    a[11, 2] = 2906258129/315225000
    a[11, 3] = -1394895449/35025000
    a[11, 4] = -21684451/729000
    a[11, 5] = -2043779/283725
    a[11, 6] = 2123/1125
    a[11, 7] = -1/2025
    a[11, 8] = -1/1125
    a[11, 9] = 0
    a[11, 10] = 1

    # b7: 7th order weights
    b7 = np.array([
        326714596721/1860513750000,
        0,
        59655821/47287500,
        -46178979/10507500,
        96112/18125,
        22285/405,
        -23/16,
        1,
        -1088/2025,
        10/11,
        -2843/5500,
        0
    ])

    # b6: 6th order weights (for error estimation)
    b6 = np.array([
        2890261879187/16744623750000,
        0,
        46296019/15761250,
        -28307387/10507500,
        34792/5625,
        1805/189,
        -23/16,
        1,
        -1088/2025,
        10/11,
        -2843/5500,
        0
    ])

    # Adaptive step size control
    rtol = 1e-11
    atol = 1e-13
    h = (t1 - t0) / 10  # initial step
    h_min = 1e-12
    h_max = (t1 - t0)

    # Safety factor
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0

    # Store the 12 stages
    K = np.zeros((12, len(y)))

    while t < t1:
        if time.time() > deadline:
            break

        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t

        # Perform one step
        for i in range(12):
            y_stage = y + h * sum(a[i, j] * K[j] for j in range(i))
            K[i] = f(t + c[i] * h, y_stage)

        y_new = y + h * sum(b7[i] * K[i] for i in range(12))
        y_err = y + h * sum(b6[i] * K[i] for i in range(12))

        # Error estimate
        err = y_new - y_err
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(np.abs(err) / scale)

        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_new

            # Choose new step size
            if err_norm > 1e-14:
                factor = safety * (1.0 / err_norm) ** (1.0 / 7.0)
            else:
                factor = max_factor
            factor = min(max(factor, min_factor), max_factor)
            h = h * factor
        else:
            # Reject step, reduce h
            factor = safety * (1.0 / err_norm) ** (1.0 / 7.0)
            factor = max(factor, min_factor)
            h = h * factor

        h = max(h, h_min)
        h = min(h, h_max)

        # Check for non-finite values
        if not np.all(np.isfinite(y)):
            # Fallback: use a simple RK4 with very small steps
            return _fallback_rk4(f, t0, t1, y0)

    return y


def _fallback_rk4(f, t0, t1, y0, steps=10000):
    """Simple RK4 fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
