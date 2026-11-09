# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    sol = scipy.integrate.solve_ivp(
        f,
        (t0, t1),
        y0,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
        max_step=(t1 - t0) / 50.0
    )

    if not sol.success:
        # Fallback to a smaller-step DOP853 if anything went wrong
        sol = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 100.0
        )

    y_end = sol.y[:, -1]

    # Ensure finite
    if not np.all(np.isfinite(y_end)):
        # Last resort: very conservative explicit RK8 with tiny steps
        steps = 20000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        # Simple RK4 as a safe fallback (not ideal but finite)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        y_end = y

    return y_end
# EVOLVE-BLOCK-END
