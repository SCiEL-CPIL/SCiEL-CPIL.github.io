import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use tight tolerances to minimize rel error
    # DOP853 is the same method as reference, but tighter tol means we are closer to true solution
    # Reference used rtol=1e-11, atol=1e-13.
    # If we use rtol=1e-12, our error is smaller.
    # rel = ||y_mine - y_ref|| / ||y_ref||
    # y_ref has error ~ 1e-11.
    # If y_mine is exact, rel ~ 1e-11.
    # If y_mine has error 1e-12, rel is still dominated by y_ref's error, so ~1e-11.
    # So pushing tol tighter than 1e-11 might not reduce rel further significantly.
    # However, it ensures we don't add *more* error.
    # Let's use rtol=1e-11, atol=1e-13 to match reference exactly?
    # If we match reference exactly, rel might be very small (numerical noise).
    # But calls might be similar.
    # Let's try rtol=1e-12 to be safe on the accuracy side.
    
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
    if not sol.success:
        # Fallback if it fails, though unlikely
        # Try looser tolerances
        sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-10, atol=1e-12)
        if not sol.success:
             raise RuntimeError("Integration failed")
             
    return sol.y[:, -1]
