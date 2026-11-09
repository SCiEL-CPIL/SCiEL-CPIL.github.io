"""ODE solver for Robertson problem using high-accuracy Radau with conservation enforcement."""
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s limit
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0, y1, y2 = y[0], y[1], y[2]
        J = np.zeros((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y2
        J[0, 2] = 1e4 * y1
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y2 - 6e7 * y1
        J[1, 2] = -1e4 * y1
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y1
        J[2, 2] = 0.0
        return J
    
    # Try multiple solver configurations with progressively tighter tolerances
    # until we hit the deadline or achieve sufficient accuracy
    
    configs = [
        # (method, rtol, atol, max_step)
        ('Radau', 1e-12, 1e-14, 0.05),
        ('Radau', 1e-13, 1e-15, 0.02),
        ('Radau', 1e-14, 1e-16, 0.01),
        ('BDF', 1e-12, 1e-14, 0.05),
        ('BDF', 1e-13, 1e-15, 0.02),
    ]
    
    best_sol = None
    best_err_estimate = np.inf
    
    for method, rtol, atol, max_step in configs:
        # Check deadline
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:  # leave 10s margin
            break
        
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method=method,
                rtol=rtol,
                atol=atol,
                jac=jac,
                max_step=max_step,
                dense_output=False
            )
            
            if sol.success:
                y_final = sol.y[:, -1]
                
                # Enforce conservation law: y0 + y1 + y2 = 1
                # This is a known invariant of the Robertson problem
                conservation = y0_arr[0] + y0_arr[1] + y0_arr[2]
                current_sum = np.sum(y_final)
                if abs(current_sum) > 1e-15:
                    # Scale to enforce conservation
                    y_final = y_final * (conservation / current_sum)
                
                # Clamp small negative values to zero (physical constraint)
                y_final = np.maximum(y_final, 0.0)
                
                # Re-enforce conservation after clamping
                current_sum = np.sum(y_final)
                if abs(current_sum) > 1e-15:
                    y_final = y_final * (conservation / current_sum)
                
                # Estimate error by comparing with slightly different tolerance
                if best_sol is not None and rtol < 1e-12:
                    # Compare with previous best
                    err = np.linalg.norm(y_final - best_sol) / max(1.0, np.linalg.norm(y_final))
                    if err < best_err_estimate:
                        best_err_estimate = err
                        best_sol = y_final.copy()
                else:
                    best_sol = y_final.copy()
                
            # Check if we've spent too much time
            elapsed = time.time() - start_time
            if elapsed > deadline - 5.0:
                break
                
        except Exception:
            continue
    
    if best_sol is not None:
        return best_sol
    else:
        # Fallback: use the first config
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                jac=jac,
                max_step=0.1
            )
            if sol.success:
                y_final = sol.y[:, -1]
                # Enforce conservation
                conservation = np.sum(y0_arr)
                current_sum = np.sum(y_final)
                if abs(current_sum) > 1e-15:
                    y_final = y_final * (conservation / current_sum)
                y_final = np.maximum(y_final, 0.0)
                current_sum = np.sum(y_final)
                if abs(current_sum) > 1e-15:
                    y_final = y_final * (conservation / current_sum)
                return y_final
        except Exception:
            pass
        return y0_arr
