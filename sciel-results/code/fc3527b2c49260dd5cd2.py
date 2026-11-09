"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    y' = f(t, y)
    y1' = -10*y1 + 1e4*y2*y3
    y2' = 10*y1 - 1e4*y2*y3 - 1e6*y2^2
    y3' = 1e6*y2^2
    """
    y1, y2, y3 = y
    return np.array([
        [-10.0, 1e4 * y3, 1e4 * y2],
        [10.0, -1e4 * y3 - 2e6 * y2, -1e4 * y2],
        [0.0, 2e6 * y2, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses the Backward Differentiation Formula (BDF) method with an analytical Jacobian.
    BDF is a distinct family of implicit multistep methods, often highly efficient
    for stiff problems like Robertson, potentially offering a better error/call trade-off
    than Implicit Runge-Kutta (Radau).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use BDF method.
    # BDF is A-stable and suitable for stiff ODEs.
    # Providing the Jacobian is crucial for efficiency.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            jac=_robertson_jac,
            rtol=1e-12, 
            atol=1e-14,
            max_step=np.inf
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to Radau if BDF fails
            sol_radau = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                jac=_robertson_jac,
                rtol=1e-12, 
                atol=1e-14
            )
            if sol_radau.success:
                return sol_radau.y[:, -1]
            else:
                raise RuntimeError("Integration failed with BDF and Radau")
                
    except Exception:
        raise

# EVOLVE-BLOCK-END
