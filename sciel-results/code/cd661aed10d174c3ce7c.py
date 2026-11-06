import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    start_time = time.time()
    # 330s limit, leave 30s margin
    deadline = start_time + 270.0
    
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    # Strategy: Use LSODA with very tight tolerances.
    # LSODA automatically switches between stiff (BDF) and non-stiff (Adams) methods.
    # For Robertson problem, it can use Adams during the slow manifold phase
    # and BDF during the fast transient, potentially reducing function calls
    # compared to fixed stiff solvers while maintaining high accuracy.
    # rtol=1e-14, atol=1e-16 for maximum precision.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='LSODA',
            rtol=1e-14,
            atol=1e-16,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Fallback to Radau if LSODA failed
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Fallback to BDF
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Last resort: Explicit RK45 (only if implicit methods failed)
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    return y0
