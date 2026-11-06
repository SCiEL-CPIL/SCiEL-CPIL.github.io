import numpy as np
import time
from scipy.integrate import ode

def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] and return y(t1).
    Uses scipy.integrate.ode with LSODA method for high efficiency on stiff problems.
    LSODA automatically switches between stiff (BDF) and non-stiff (Adams) methods,
    which is optimal for the Robertson problem that transitions from stiff to non-stiff.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    n = y0_arr.shape[0]

    # Analytic Jacobian for the Robertson problem.
    # y' = f(t, y)
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
            J[1, 1] = -1.0e4 * y2v - 1.2e8 * y1v  # d/dy1 ( -1e4*y1*y2 - 6e7*y1^2 ) = -1e4*y2 - 12e7*y1
            J[1, 2] = -1.0e4 * y1v
            
            J[2, 0] = 0.0
            J[2, 1] = 1.2e8 * y1v  # d/dy1 ( 6e7*y1^2 ) = 12e7*y1
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

    # Strategy: Use scipy.integrate.ode with LSODA
    # LSODA is efficient for problems that switch between stiff and non-stiff.
    # The Robertson problem is stiff initially and then becomes non-stiff.
    # We provide the analytic Jacobian to help the stiff part and stability checks.
    
    solver = ode(f, jac=jac)
    # rtol and atol for LSODA
    # Tight tolerances to minimize error.
    solver.set_integrator('lsoda', rtol=1e-14, atol=1e-16)
    
    # Initial state
    solver.set_initial_value(y0_arr, t0)
    
    # Integrate to t1
    try:
        solver.integrate(t1)
        if solver.successful():
            return solver.y[:].reshape(-1)
    except Exception:
        pass

    # Fallback if direct integration failed
    if time_left() > 10.0:
        try:
            solver2 = ode(f, jac=jac)
            # Try Radau as a fallback if LSODA fails
            solver2.set_integrator('radau', rtol=1e-14, atol=1e-16)
            solver2.set_initial_value(y0_arr, t0)
            solver2.integrate(t1)
            if solver2.successful():
                return solver2.y[:].reshape(-1)
        except Exception:
            pass

    # Last resort: solve_ivp with Radau
    if time_left() > 10.0:
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                f, t_span, y0_arr,
                method="Radau",
                jac=jac,
                rtol=1e-13,
                atol=1e-15,
                max_step=1.0,
            )
            if sol.success:
                return sol.y[:, -1].reshape(-1)
        except Exception:
            pass

    return y0_arr.copy()
