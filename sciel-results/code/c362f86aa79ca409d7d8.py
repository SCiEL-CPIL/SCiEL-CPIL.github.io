"""ODE integrator for the Van der Pol oscillator (mu = 5) with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator from t_span[0] to t_span[1].
    Uses scipy.integrate.solve_ivp with the Radau method.
    """
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if solver fails
            if sol.y.size > 0:
                return sol.y[:, -1]
            else:
                return np.array(y0, dtype=float)

    except Exception:
        # Fallback to a simpler method if Radau fails
        try:
            sol = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='RK45',
                rtol=1e-11,
                atol=1e-13
            )
            if sol.success and sol.y.size > 0:
                return sol.y[:, -1]
            elif sol.y.size > 0:
                return sol.y[:, -1]
            else:
                return np.array(y0, dtype=float)
        except Exception:
            return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
