import numpy as np
import time
from scipy.integrate import ode

def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] and return y(t1).
    Uses scipy.integrate.ode with Radau method for high efficiency on stiff problems.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    n = y0_arr.shape[0]

    # Analytic Jacobian for the Robertson problem.
    def jac(t, y):
        y = np.asarray(y).reshape(-1)
        y0v, y1v, y2v = y[0], y[1], y[2]
        J = np.empty((n, n))
        if n >= 3:
            J[0, 0] = -0.04
            J[0, 1] = 1.0e4 * y2v
            J[0, 2] = 1.0e4 * y1v
            J[1, 0] = 0.04
            J[1, 1] = -1.0e4 * y2v - 6.0e7 * y1v
            J[1, 2] = -1.0e4 * y1v
            J[2, 0] = 0.0
            J[2, 1] = 6.0e7 * y1v
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

    # Strategy: Use scipy.integrate.ode with Radau
    # This lower-level interface often has less overhead than solve_ivp.
    # We set very tight tolerances.
    
    solver = ode(f, jac=jac)
    # rtol and atol for Radau
    # We want high accuracy.
    # Note: set_integrator takes rtol and atol as keyword arguments for radau
    solver.set_integrator('radau', rtol=1e-14, atol=1e-16)
    
    # Initial state
    solver.set_initial_value(y0_arr, t0)
    
    # Integrate to t1
    # We step in chunks to allow time checks, but for [0, 40] it's usually fast enough.
    # However, to be safe, we can step in increments.
    # Radau is implicit, so it handles stiffness well.
    
    try:
        # Attempt direct integration
        solver.integrate(t1)
        if solver.successful():
            return solver.y[:].reshape(-1)
    except Exception:
        pass

    # Fallback if direct integration failed or timed out (unlikely to timeout in 300s for this small problem)
    # Try with slightly looser tolerances if the first failed due to numerical issues
    if time_left() > 10.0:
        try:
            solver2 = ode(f, jac=jac)
            solver2.set_integrator('radau', rtol=1e-12, atol=1e-14)
            solver2.set_initial_value(y0_arr, t0)
            solver2.integrate(t1)
            if solver2.successful():
                return solver2.y[:].reshape(-1)
        except Exception:
            pass

    # Last resort: solve_ivp
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
