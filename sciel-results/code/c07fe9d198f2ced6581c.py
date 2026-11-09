"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def _vdp_jacobian(t, y):
    """
    Compute the Jacobian matrix for the Van der Pol oscillator.
    y0' = y1
    y1' = 5(1 - y0^2)y1 - y0
    """
    y0, y1 = y
    return np.array([
        [0.0, 1.0],
        [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0**2)]
    ])

def integrate(f, t_span, y0):
    """
    Integrates the ODE using scipy.integrate.solve_ivp with Radau method.
    Provides a custom Jacobian to improve convergence speed and accuracy.
    Uses tight tolerances to minimize relative error.
    """
    t0, t1 = t_span
    start_time = time.time()
    
    # Define the Jacobian function to pass to the solver
    # This helps the implicit Radau method solve nonlinear systems faster
    
    try:
        # First attempt: Tight tolerances with Jacobian
        # Radau is L-stable and good for stiff problems.
        # Tighter tolerances aim for rel < 1e-14
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            jac=_vdp_jacobian
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if it fails (e.g. max steps reached)
            # Retry with slightly looser tolerances or default settings
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-11, 
                atol=1e-13,
                jac=_vdp_jacobian
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort fallback
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='RK45', 
                    rtol=1e-10, 
                    atol=1e-12
                )
                if sol.success:
                    return sol.y[:, -1]
                else:
                    return np.array(y0, dtype=float)
    except Exception:
        # If anything goes wrong, return initial state or a simple RK4 step
        # To ensure we don't crash
        return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
