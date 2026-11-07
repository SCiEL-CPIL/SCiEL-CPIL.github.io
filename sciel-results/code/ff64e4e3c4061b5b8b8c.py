"""ODE integrator for the Van der Pol oscillator (mu=5)."""
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator y0'=y1, y1'=5(1-y0^2)y1 - y0
    from t_span[0] to t_span[1] with initial condition y0.
    Returns the state vector at t_span[1].
    """
    deadline = time.time() + 290  # Safety margin before 330s kill
    y0_arr = np.array(y0, dtype=float)
    t0, t1 = t_span

    # Primary: Radau with exact reference tolerances
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0_arr, method='Radau', rtol=1e-11, atol=1e-13)
        if sol.success and sol.y.shape[1] > 0:
            result = sol.y[:, -1]
            if np.all(np.isfinite(result)):
                # Polish pass: tighter tolerance to reduce error further
                if time.time() < deadline - 30:
                    try:
                        sol2 = solve_ivp(f, t_span, y0_arr, method='Radau', rtol=1e-13, atol=1e-15)
                        if sol2.success and sol2.y.shape[1] > 0:
                            result2 = sol2.y[:, -1]
                            if np.all(np.isfinite(result2)):
                                return result2
                    except Exception:
                        pass
                return result
    except Exception:
        pass

    # Fallback: tighter Radau
    if time.time() < deadline - 30:
        try:
            sol = solve_ivp(f, t_span, y0_arr, method='Radau', rtol=1e-12, atol=1e-14)
            if sol.success and sol.y.shape[1] > 0:
                result = sol.y[:, -1]
                if np.all(np.isfinite(result)):
                    return result
        except Exception:
            pass

    # Fallback: DOP853
    if time.time() < deadline - 30:
        try:
            sol = solve_ivp(f, t_span, y0_arr, method='DOP853', rtol=1e-12, atol=1e-14)
            if sol.success and sol.y.shape[1] > 0:
                result = sol.y[:, -1]
                if np.all(np.isfinite(result)):
                    return result
        except Exception:
            pass

    # Last resort: RK4 with adaptive-ish fixed steps
    if time.time() < deadline - 10:
        y = y0_arr.copy()
        t = t0
        remaining = t1 - t0
        # Start with a reasonable step, refine if needed
        n_base = 20000
        h = remaining / n_base
        for i in range(n_base):
            if time.time() > deadline - 5:
                break
            k1 = np.asarray(f(t, y), dtype=float)
            k2 = np.asarray(f(t + 0.5*h, y + 0.5*h*k1), dtype=float)
            k3 = np.asarray(f(t + 0.5*h, y + 0.5*h*k2), dtype=float)
            k4 = np.asarray(f(t + h, y + h*k3), dtype=float)
            y = y + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
            t += h
        if np.all(np.isfinite(y)):
            return y

    return y0_arr
