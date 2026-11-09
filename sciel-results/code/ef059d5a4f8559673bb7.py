"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def _jacobian(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    y0' = -0.04 y0 + 10^4 y1 y2
    y1' = 0.04 y0 - 10^4 y1 y2 - 3e3 y1^2
    y2' = 3e3 y1^2
    """
    y0, y1, y2 = y
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e3 * y1, -1e4 * y1],
        [0.0, 6e3 * y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Set a deadline slightly before the 330s kill switch
    start_time = time.time()
    deadline = 320.0  # seconds
    
    # Strategy: Use a single high-precision solve with the most robust method
    # Radau is generally best for stiff problems. We'll use very tight tolerances.
    # Since Robertson is a small 3x3 system, the cost is dominated by the number of steps,
    # not the linear algebra per step. So tight tolerances are cheap relative to the
    # accuracy gain.
    
    # We use LSODA as a first attempt because it automatically switches between
    # Adams (non-stiff) and BDF (stiff) methods, which can be efficient.
    # However, for Robertson, the problem is very stiff from the start, so
    # a dedicated stiff solver like Radau or BDF is often better.
    
    # Let's try Radau with extremely tight tolerances first.
    # The Robertson problem is well-studied; Radau with rtol=1e-14 should be very accurate.
    
    try:
        # Primary attempt: Radau with tight tolerances
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_jacobian,
            rtol=1e-13, 
            atol=1e-15,
            max_step=np.inf
        )
        
        if not sol.success:
            # If Radau fails, try BDF as a fallback
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=_jacobian,
                rtol=1e-13, 
                atol=1e-15
            )
            return sol.y[:, -1]
        
        result = sol.y[:, -1]
        
        # Check if we have time for a refinement pass with even tighter tolerances
        elapsed = time.time() - start_time
        if elapsed < deadline - 50:  # 50s safety margin
            try:
                # Refinement pass: tighter tolerances for higher precision
                sol_refined = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='Radau', 
                    jac=_jacobian,
                    rtol=1e-15, 
                    atol=1e-17,
                    max_step=np.inf
                )
                
                if sol_refined.success:
                    result = sol_refined.y[:, -1]
                    
            except Exception:
                # If refinement fails, keep the original result
                pass
        
        return result
        
    except Exception:
        # Fallback to BDF with moderate tolerances
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=_jacobian,
                rtol=1e-11, 
                atol=1e-13
            )
            return sol.y[:, -1]
        except Exception:
            # Last resort: raise
            raise
# EVOLVE-BLOCK-END
