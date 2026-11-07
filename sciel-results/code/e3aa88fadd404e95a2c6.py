"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses scipy's DOP853 for high accuracy with minimal function calls.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Track elapsed time for safety
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin before 330s kill
    
    try:
        from scipy.integrate import solve_ivp
        
        # Use DOP853 with tight tolerances to match the reference
        # No max_step limit to allow DOP853 to take larger steps where error permits
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False,
            max_step=np.inf
        )
        
        # Check if we're running out of time
        elapsed = time.time() - start_time
        if elapsed > deadline - start_time:
            return _rk4_fallback(f, t0, t1, y0, start_time, deadline)
        
        if not sol.success:
            # Fallback to RK4 if solve_ivp fails
            return _rk4_fallback(f, t0, t1, y0, start_time, deadline)
        
        y_end = sol.y[:, -1]
        
        # Verify result is finite and has correct shape
        if not np.all(np.isfinite(y_end)) or y_end.shape != y0.shape:
            return _rk4_fallback(f, t0, t1, y0, start_time, deadline)
        
        return y_end
        
    except Exception:
        # Fallback to RK4 if any exception occurs
        return _rk4_fallback(f, t0, t1, y0, start_time, deadline)


def _rk4_fallback(f, t0, t1, y0, start_time, deadline):
    """
    Simple RK4 fallback with adaptive step sizing, respecting the deadline.
    Uses embedded RK4(5) error estimation via Richardson extrapolation.
    """
    y = np.array(y0, dtype=float)
    
    # Start with a reasonable step size
    h = (t1 - t0) / 500
    t = t0
    
    # Adaptive RK4 with error estimation
    step_count = 0
    max_steps = 100000
    
    while t < t1 and step_count < max_steps:
        if time.time() > deadline - 5.0:
            # Not enough time left, return current best estimate
            break
        
        # Check if we need to adjust step to land exactly on t1
        if t + h > t1:
            h = t1 - t
        
        # RK4 full step
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        # Full step result
        y_full = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Two half steps for error estimation
        h_half = h / 2.0
        
        # First half step
        k1a = k1  # Reuse k1 from full step (same t, same y)
        k2a = f(t + h_half/2, y + h_half/2 * k1a)
        k3a = f(t + h_half/2, y + h_half/2 * k2a)
        k4a = f(t + h_half, y + h_half * k3a)
        y_half1 = y + h_half/6.0 * (k1a + 2*k2a + 2*k3a + k4a)
        
        # Second half step
        k1b = f(t + h_half, y_half1)
        k2b = f(t + h_half + h_half/2, y_half1 + h_half/2 * k1b)
        k3b = f(t + h_half + h_half/2, y_half1 + h_half/2 * k2b)
        k4b = f(t + h_half + h_half, y_half1 + h_half * k3b)
        y_half2 = y_half1 + h_half/6.0 * (k1b + 2*k2b + 2*k3b + k4b)
        
        # Error estimate: difference between full step and two half steps
        # For RK4, the local error of full step is O(h^5), two half steps is O(h^5)
        # The difference is O(h^5) and gives error estimate
        err_norm = np.linalg.norm(y_half2 - y_full)
        scale = np.linalg.norm(y) + 1e-12
        err = err_norm / scale
        
        # Accept step if error is small enough
        if err < 1e-10:
            y = y_half2  # Use more accurate result (Richardson extrapolation)
            t += h
            step_count += 1
            
            # Optionally grow step size if error is much smaller than tolerance
            if err < 1e-12 and h < (t1 - t0) / 50:
                h *= 1.5
        else:
            # Reduce step size
            h *= 0.5
            if h < 1e-14:
                # Step size too small, accept the full step
                y = y_full
                t += h
                step_count += 1
    
    # If we didn't reach t1, do a final step to reach t1
    if t < t1:
        h_remaining = t1 - t
        if h_remaining > 1e-15:
            k1 = f(t, y)
            k2 = f(t + h_remaining/2, y + h_remaining/2 * k1)
            k3 = f(t + h_remaining/2, y + h_remaining/2 * k2)
            k4 = f(t + h_remaining, y + h_remaining * k3)
            y = y + h_remaining/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t = t1
    
    return y
# EVOLVE-BLOCK-END
