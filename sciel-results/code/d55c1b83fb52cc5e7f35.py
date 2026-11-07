"""High-precision ODE integrator using SciPy's DOP853 to match reference behavior."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1].
    Uses scipy.integrate.solve_ivp with DOP853 method to match the reference
    solution's accuracy and call count profile.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use the same solver settings as the reference to minimize relative error
    # and keep function calls low.
    sol = scipy.integrate.solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback in case of unexpected failure, though unlikely for this problem
        # Try a standard RK45 with tight tolerance
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-12,
            atol=1e-14
        )
        if not sol.success:
            raise RuntimeError("Integration failed")

    return sol.y[:, -1]
# EVOLVE-BLOCK-END
