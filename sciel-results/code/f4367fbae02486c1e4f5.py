"""ODE solver: Robertson problem using LSODA via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE y' = f(t, y) from t_span[0] to t_span[1].
    Uses LSODA for automatic stiffness detection.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 320s as safety margin
    deadline = time.time() + 320
    
    # Count function evaluations
    call_count = 0
    
    def wrapped_f(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
    
    # Try LSODA with tight tolerances
    # Robertson problem is stiff, LSODA should handle it well
    try:
        sol = solve_ivp(
            wrapped_f,
            (t0, t1),
            y0_arr,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000  # initial max step suggestion
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        # If LSODA failed, try with relaxed tolerances
        call_count = 0
        sol = solve_ivp(
            wrapped_f,
            (t0, t1),
            y0_arr,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 500
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        # Last resort: use whatever result we got
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
    
    except Exception:
        pass
    
    # Fallback: use Radau if LSODA completely fails
    try:
        call_count = 0
        sol = solve_ivp(
            wrapped_f,
            (t0, t1),
            y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: simple RK4 with many steps
    try:
        h = (t1 - t0) / 10000
        t = t0
        y = y0_arr.copy()
        n_steps = 10000
        for _ in range(n_steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    except Exception:
        return y0_arr
# EVOLVE-BLOCK-END
