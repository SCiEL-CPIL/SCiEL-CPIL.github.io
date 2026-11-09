"""
Strategy:
Use the roots of the generalized Laguerre polynomial L_12^(-1/2)(x) as the double-root locations.
These nodes are optimal for interpolation in the Laguerre basis, which minimizes the magnitude
of the quotient q(x), thereby minimizing the largest real root B.
"""
import time
import math
import sympy

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Uses the roots of L_12^(-1/2)(x) as the primary candidate.
    """
    t0 = time.time()
    DEADLINE = 600
    SAFETY_MARGIN = 30
    
    # Deadline guard
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        return [4.0 * (i + 1) for i in range(12)]

    try:
        # Compute roots of L_12^(-1/2)(x)
        x = sympy.Symbol('x')
        # alpha = -1/2
        L_poly = sympy.genlaguerre(12, sympy.Rational(-1, 2), x)
        roots = L_poly.roots()
        
        # Convert to floats and sort
        z_list = sorted([float(r) for r in roots])
        
        # Validate
        if len(z_list) == 12:
            if all(zi > 0 for zi in z_list):
                if len(set(z_list)) == 12:
                    return z_list
    except Exception:
        pass
    
    # Fallback to seed if sympy fails or roots are invalid
    return [4.0 * (i + 1) for i in range(12)]
