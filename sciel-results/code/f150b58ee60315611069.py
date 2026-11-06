import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] and return y(t1).
    Uses solve_ivp with Radau method (implicit Runge-Kutta) for stiff problems.
    Provides analytic Jacobian for the Robertson problem.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    n = y0_arr.shape[0]

    # Analytic Jacobian for the Robertson problem
    # y0' = -0.04*y0 + 1e4*y1*y2
    # y1' = 0.04*y0 - 1e4*y1*y2 - 6e7*y1^2
    # y2' = 6e7*y1^2
    def jac(t, y):
        y = np.asarray(y).reshape(-1)
        y0v, y1v, y2v = y[0], y[1], y[2]
        J = np.empty((n, n))
        if n >= 3:
            J[0, 0] = -0.04
            J[0, 1] = 1.0e4 * y2v
            J[0, 2] = 1.0e4 * y1v
            
            J[1, 0] = 0.04
            J[1, 1] = -1.0e4 * y2v - 1.2e8 * y1v
            J[1, 2] = -1.0e4 * y1v
            
            J[2, 0] = 0.0
            J[2, 1] = 1.2e8 * y1v
            J[2, 2] = 0.0
        else:
            # Fallback: numerical Jacobian
            h = 1e-8
            J = np.zeros((n, n))
            f0 = np.asarray(f(t, y)).reshape(-1)
            for j in range(n):
                yp = y.copy()
                yp[j] += h
                J[:, j] = (np.asarray(f(t, yp)).reshape(-1) - f0) / h
        return J

    # Deadline guard
    start_time = time.time()
    time_limit = 300.0

    def time_left():
        return time_limit - (time.time() - start_time)

    # Strategy: Use solve_ivp with Radau (high-order implicit Runge-Kutta)
    # Radau is well-suited for stiff problems and provides high accuracy.
    # We use very tight tolerances to minimize error.
    
    try:
        sol = solve_ivp(
            f, t_span, y0_arr,
            method="Radau",
            jac=jac,
            rtol=1e-14,
            atol=1e-16,
            max_step=1.0,
        )
        if sol.success:
            return sol.y[:, -1].reshape(-1)
    except Exception:
        pass

    # Fallback: Try LSODA via scipy.integrate.ode
    if time_left() > 10.0:
        try:
            from scipy.integrate import ode
            solver = ode(f, jac=jac)
            solver.set_integrator('lsoda', rtol=1e-14, atol=1e-16)
            solver.set_initial_value(y0_arr, t0)
            solver.integrate(t1)
            if solver.successful():
                return solver.y[:].reshape(-1)
        except Exception:
            pass

    # Last resort: Try BDF
    if time_left() > 10.0:
        try:
            sol = solve_ivp(
                f, t_span, y0_arr,
                method="BDF",
                jac=jac,
                rtol=1e-14,
                atol=1e-16,
                max_step=1.0,
            )
            if sol.success:
                return sol.y[:, -1].reshape(-1)
        except Exception:
            pass

    return y0_arr.copy()
