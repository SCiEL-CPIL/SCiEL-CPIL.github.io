"""ODE solver for convection-diffusion 100-point problem using matrix exponential strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import scipy.integrate
import time

def _compute_jacobian(f, t, y, eps=1e-7):
    """Compute Jacobian using forward differences."""
    n = len(y)
    J = np.zeros((n, n))
    f0 = f(t, y)
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps * max(1.0, abs(y[j]))
        J[:, j] = (f(t, y + e) - f0) / e[j]
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin from 330s limit
    
    # 1. Compute Jacobian at t0 to probe linearity
    J = _compute_jacobian(f, t0, y)
    
    # Strategy: Try exact matrix exponential for linear autonomous system
    # For a linear system y' = A*y, the exact solution is y(t) = exp(A*(t-t0)) * y0
    # This requires only n+1 function calls (1 for f0 in Jacobian, n for columns)
    # and achieves machine precision.
    
    def try_linear_exponential():
        nonlocal y, t0, t1, J, deadline, start_time
        
        # Check deadline
        if time.time() > deadline:
            return None
            
        # Compute exp(A * (t1 - t0))
        # For n=100, this is a 100x100 matrix exponential
        T = t1 - t0
        exp_AT = scipy.linalg.expm(J * T)
        
        # Apply to initial condition
        y_final = exp_AT @ y
        
        # Check for numerical stability
        if not np.all(np.isfinite(y_final)) or np.max(np.abs(y_final)) > 1e12:
            return None
            
        # Verify linearity assumption by checking residual at final point
        # If the system is truly linear, f(t1, y_final) should equal J @ y_final
        f_val = f(t1, y_final)
        residual = np.max(np.abs(f_val - J @ y_final))
        
        # Heuristic: if residual is too large, the linear model is invalid
        scale = np.max(np.abs(y_final)) + 1.0
        if residual > 1e-10 * scale:
            # Linear assumption failed, abort
            return None
            
        return y_final

    # Attempt 1: Exact exponential method for linear systems
    y_result = try_linear_exponential()
    
    if y_result is not None:
        return y_result

    # Fallback: Use scipy.integrate.solve_ivp with Radau (implicit, stiff)
    def solve_ivp_fallback():
        # Use Radau method with tight tolerances
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 5.0
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Try BDF
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14,
                max_step=(t1 - t0) / 5.0
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: RK45
                sol = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='RK45',
                    rtol=1e-12,
                    atol=1e-14
                )
                if sol.success:
                    return sol.y[:, -1]
                raise RuntimeError("All solvers failed")

    y_result = solve_ivp_fallback()
    return y_result
# EVOLVE-BLOCK-END
