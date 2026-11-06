"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    """Main integration function using scipy for high accuracy with low call count."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 330s kill

    # Primary: DOP853 (8th order) without restrictive max_step
    # For smooth convection-diffusion, DOP853 takes large steps efficiently
    result = scipy.integrate.solve_ivp(
        f, t_span, y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        # No max_step restriction - let the adaptive method choose optimal steps
    )

    if result.success and result.t[-1] == t1:
        y_primary = result.y[:, -1]
        
        # Check if we have time for a polish pass with tighter tolerances
        elapsed = time.time() - start_time
        if elapsed < 250.0:
            # Polish pass: even tighter tolerances to squeeze out more accuracy
            result2 = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='DOP853',
                rtol=1e-14,
                atol=1e-16,
            )
            if result2.success and result2.t[-1] == t1:
                return result2.y[:, -1]
        
        return y_primary

    # Fallback: use Radau for stiff problems
    if time.time() > deadline - 10.0:
        if result.y is not None and len(result.y) > 0:
            return result.y[:, -1]
        return y0

    result = scipy.integrate.solve_ivp(
        f, t_span, y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14
    )

    if result.success and result.t[-1] == t1:
        return result.y[:, -1]

    # Check time before RK4 fallback
    if time.time() > deadline - 5.0:
        if result.y is not None and len(result.y) > 0:
            return result.y[:, -1]
        return y0

    # Last resort: fixed-step RK4 with time check
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for i in range(steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

        # Check time periodically
        if i % 1000 == 0 and time.time() > deadline - 2.0:
            break

    return y
# EVOLVE-BLOCK-END
