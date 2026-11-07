# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 180.0  # safety margin under 210s kill

def _g_coeffs(c):
    """
    Returns coefficients of g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12.
    Hermite polynomials (physicists):
    H0 = 1
    H4 = 16x^4 - 48x^2 + 12
    H8 = 256x^8 - 4480x^6 + 24192x^4 - 40320x^2 + 1680
    H12 = 4096x^12 - 147456x^10 + 1684608x^8 - 9067200x^6 + 20901120x^4 - 20321280x^2 + 665280
    
    Coeffs array is [a12, a10, a8, a6, a4, a2, a0] (odd powers are 0)
    """
    c0, c1, c2 = c
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    # c3 = -(c0*1 + c1*12 + c2*1680) / 665280
    c3 = -(c0 + 12.0 * c1 + 1680.0 * c2) / 665280.0
    
    # Construct polynomial coefficients in descending order of power
    # Powers: 12, 10, 8, 6, 4, 2, 0
    
    a12 = 4096.0 * c3
    a10 = -147456.0 * c3
    a8 = 1684608.0 * c3 + 256.0 * c2
    a6 = -9067200.0 * c3 - 4480.0 * c2
    a4 = 20901120.0 * c3 + 24192.0 * c2 - 48.0 * c1
    a2 = -20321280.0 * c3 - 40320.0 * c2 + 12.0 * c1
    a0 = 665280.0 * c3 + 1680.0 * c2 + 12.0 * c1 + c0
    
    # The polynomial is a12*x^12 + a10*x^10 + ... + a0
    # np.roots expects [a_n, ..., a_0]
    return np.array([a12, 0, a10, 0, a8, 0, a6, 0, a4, 0, a2, 0, a0])

def _objective(c):
    """
    Returns the largest real root r of g(x) where g changes sign.
    This is equivalent to the largest real root of q(x) where q changes sign.
    If no such root exists, return a large penalty.
    """
    try:
        coeffs = _g_coeffs(c)
        roots = np.roots(coeffs)
        
        # Filter real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-9:
                real_roots.append(r.real)
        
        if not real_roots:
            return 1e6
            
        # We need to check sign change at each real root.
        # Since g is a polynomial, we can evaluate g at r +/- epsilon.
        # However, numerical precision might be an issue for very close roots.
        # A better check: multiplicity. If multiplicity is odd, sign changes.
        # Finding multiplicity numerically is tricky.
        # Let's use the sign change test with a small epsilon.
        
        # Sort roots to find the largest
        real_roots.sort(reverse=True)
        
        # To avoid division by zero or issues at 0, note that g(0)=0 by construction.
        # We should ignore x=0.
        
        # Define g(x)
        def g(x):
            return np.polyval(coeffs, x)
            
        # Check sign change for each root from largest to smallest
        eps = 1e-6
        for r in real_roots:
            if abs(r) < 1e-9:
                continue # Skip 0
            
            # Check sign change
            # Use a small step relative to magnitude
            step = max(1e-6, abs(r) * 1e-6)
            val_left = g(r - step)
            val_right = g(r + step)
            
            # If signs are different, it's a sign-changing root
            if val_left * val_right < 0:
                return float(r)
                
        # If no sign-changing root found, return penalty
        return 1e6
        
    except Exception:
        return 1e6

def construct():
    start = time.time()
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best_c)

    def remaining():
        return _DEADLINE - (time.time() - start)

    # Global search using Differential Evolution
    # Bounds for coefficients: 
    # c0, c1, c2 can vary. 
    # The seed is [-1.13, 0.04, 0.0003]
    # Let's try a range around the seed.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # DE is stochastic, so we need to be careful with time.
    # maxiter=100, popsize=15 might be too slow if _objective is expensive.
    # _objective calls np.roots on a degree 12 poly. This is fast.
    # Let's try DE first.
    
    try:
        if remaining() > 30:
            res_de = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                polish=False, # We will polish manually
                updateinterval=10,
                callback=lambda x, convergence: (time.time() - start) > _DEADLINE - 10
            )
            if res_de.fun < best_val:
                best_val = float(res_de.fun)
                best_c = np.array(res_de.x, dtype=float)
    except Exception:
        pass

    # Local polish with Nelder-Mead
    if remaining() > 10:
        starts = [best_c.copy()]
        # Add some perturbations
        rng = np.random.default_rng(0)
        for scale in (0.1, 0.5, 1.0):
            for _ in range(5):
                p = best_c + scale * rng.standard_normal(3)
                starts.append(p)
        
        for s in starts:
            if remaining() < 5.0:
                break
            try:
                res = minimize(
                    _objective,
                    s,
                    method="Nelder-Mead",
                    options={"maxiter": 200, "xatol": 1e-9, "fatol": 1e-12, "adaptive": True}
                )
                if remaining() < 3.0:
                    break
                if res.fun < best_val:
                    best_val = float(res.fun)
                    best_c = np.array(res.x, dtype=float)
            except Exception:
                continue

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
