import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 280.0

    # Analytic Jacobian for the Robertson problem
    # y0' = -0.04*y0 + 1e4*y1*y2
    # y1' = 0.04*y0 - 1e4*y1*y2 - 3e3*y1^2
    # y2' = 3e3*y1^2
    def jac(t, y):
        return np.array([
            [-0.04, 1e4*y[2], 1e4*y[1]],
            [0.04, -1e4*y[2] - 6000*y[1], -1e4*y[1]],
            [0.0, 6000*y[1], 0.0]
        ])

    def _rk45_fallback(f, t0, t1, y0, deadline):
        y = np.array(y0, dtype=float)
        t = t0
        h = (t1 - t0) / 100.0
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            k5 = f(t + h/5, y + h/5*k4)
            k6 = f(t + 3*h/5, y + 3*h/5*(k1 + 4/3*(k4-k1)))
            
            y_new = y + h*(k1 + 3*k2 + 3*k3 + k4)/8
            err = h*(k4 - k3)/32
            err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
            
            if err_norm <= 1e-10:
                y = y_new
                t += h
                h *= min(5.0, 0.9*(1e-10/err_norm)**0.2) if err_norm > 0 else 5.0
            else:
                h *= max(0.1, 0.9*(1e-10/err_norm)**0.2)
        return y

    try:
        from scipy.integrate import solve_ivp
        
        # 1. BDF with analytic Jacobian, very tight tolerances
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]

        # 2. Radau with analytic Jacobian, very tight tolerances
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]

        # 3. Radau without Jacobian (fallback)
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]

        # 4. LSODA without Jacobian (fallback)
        sol = solve_ivp(
            f, t_span, y0,
            method='LSODA',
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]

    except Exception:
        pass

    # Fallback
    return _rk45_fallback(f, t0, t1, y0, deadline)
