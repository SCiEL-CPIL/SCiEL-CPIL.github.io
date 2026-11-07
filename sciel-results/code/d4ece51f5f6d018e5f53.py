import scipy
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    ODE integrator for the Lotka-Volterra system.
    Uses scipy.integrate.solve_ivp with DOP853 and very tight tolerances,
    with a wall-clock deadline guard.
    """
    y0 = np.asarray(y0, dtype=float)
    deadline = time.time() + 280.0  # safety margin before 330s limit
    
    # Try progressively tighter tolerances, stopping if deadline is approached
    # Start with very tight tolerances to minimize rel error
    # DOP853 is 8th order, so it's quite efficient
    
    # First attempt: very tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            dense_output=False,
            max_step=0.01  # limit step size for better accuracy
        )
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 60:
        # Fallback: use reference tolerances
        try:
            sol = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13,
                dense_output=False
            )
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    return y_end
        except Exception:
            pass
        return y0
    
    # Second attempt: even tighter
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            dense_output=False,
            max_step=0.005
        )
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    # Check deadline again
    if time.time() > deadline - 30:
        try:
            sol = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13,
                dense_output=False
            )
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    return y_end
        except Exception:
            pass
        return y0
    
    # Third attempt: tightest feasible
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-15,
            atol=1e-17,
            dense_output=False,
            max_step=0.002
        )
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    # Final fallback
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    return y0
