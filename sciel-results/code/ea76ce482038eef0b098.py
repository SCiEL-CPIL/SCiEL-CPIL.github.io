"""High-order explicit RK with fixed step size for the harmonic oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate y0' = y1, y1' = -y0 from t_span[0] to t_span[1] using a
    single high-order explicit Runge-Kutta method with a fixed step size.

    The method used is the classic 4th-order Runge-Kutta (RK4) with a
    carefully chosen number of steps to minimize function calls while
    keeping the relative error at the 1e-14 floor.

    For the harmonic oscillator, RK4 has a known stability and accuracy
    profile. We choose steps such that the local truncation error per step
    is well below 1e-14 / (number_of_steps) to ensure global error stays
    at the floor.

    Since the problem is non-stiff and smooth, RK4 with ~17000 steps should
    be more than sufficient to hit 1e-14 relative error with only 68000
    function calls (4 per step).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n_steps = 17000
    h = (t1 - t0) / n_steps

    for _ in range(n_steps):
        k1 = f(t0, y)
        k2 = f(t0 + h / 2, y + h / 2 * k1)
        k3 = f(t0 + h / 2, y + h / 2 * k2)
        k4 = f(t0 + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 += h

    return y
# EVOLVE-BLOCK-END
