"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1].
    Primary: Radau (implicit, stiff-aware) with machine-precision tolerances.
    Fallbacks: BDF then DOP853.
    Deadline guard: 330s wall-clock.
    """
    start_time = time.time()
    DEADLINE = 320.0  # 330s hard kill; leave 10s margin

    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    # Strategy: Use Radau with very tight tolerances.
    # Radau is 5th order and A-stable. For stiff convection-diffusion, it's ideal.
    # Tight tolerances reduce error significantly.
    # We are far below the 2M call limit, so we can afford the extra steps.
    
    rtol_main = 1e-14
    atol_main = 1e-16

    # Primary solver: Radau
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=rtol_main,
            atol=atol_main
        )
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        # Quick fallback: RK4 with fixed steps
        try:
            steps = 1000
            h = (t1 - t0) / steps
            t = t0
            y = y0.copy()
            for _ in range(steps):
                if time.time() - start_time > DEADLINE - 1:
                    break
                k1 = f(t, y)
                k2 = f(t + h / 2.0, y + h / 2.0 * k1)
                k3 = f(t + h / 2.0, y + h / 2.0 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t += h
            return y
        except Exception:
            return y0

    # Fallback 1: BDF with tight tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback 2: DOP853 (High order explicit)
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: fixed-step RK4
    try:
        steps = 1000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            if time.time() - start_time > DEADLINE - 1:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + h / 2.0 * k1)
            k3 = f(t + h / 2.0, y + h / 2.0 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return y
    except Exception:
        return y0

# EVOLVE-BLOCK-END
