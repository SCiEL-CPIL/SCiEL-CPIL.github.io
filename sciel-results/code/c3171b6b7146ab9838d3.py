"""Search program for the Uncertainty Inequality task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 180.0
_SEED = np.array([-113.0 / 100.0, 0.5, 1.0 / 3240.0])
_PI = np.pi


def _q_coeffs(c0, c1, c2):
    """Return (c3, [q0..q5]) for q = g/x^2 with g = c0 H0 + c1 H4 + c2 H8 + c3 H12."""
    c3 = -(c0 * 1.0 + c1 * (-12.0) + c2 * 1680.0) / (-498960.0)
    q0 = c1 * (-6.0) + c2 * 210.0 + c3 * (-6930.0)
    q1 = c1 * 1.0 + c2 * (-28.0) + c3 * 1155.0
    q2 = c2 * 1.0 + c3 * (-66.0)
    q3 = c3
    q4 = 0.0
    q5 = 0.0
    return c3, np.array([q0, q1, q2, q3, q4, q5])


def _compute_value(c0, c1, c2):
    """
    Returns (value, is_valid).
    value = r^2 / (2*pi) where r is the largest sign-changing root.
    is_valid is True if q(x) > 0 for all x > r (approximated by grid check).
    """
    c3, q = _q_coeffs(c0, c1, c2)
    if c3 <= 0:
        return np.inf, False

    # Find roots
    roots = np.roots(q)
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
    pos_roots = real_roots[real_roots > 1e-8]
    
    if len(pos_roots) == 0:
        # If no positive roots, and q(0) > 0 and c3 > 0, then q is always positive?
        # If q(0) < 0, there must be a positive root.
        if np.polyval(q, 0) > 0:
            return 0.0, True # r=0 is the limit? No, r must be a root. 
            # If no positive roots, the "largest root" logic is tricky. 
            # Usually we want a specific r. If r->0, value->0.
            # But g(0)=0 implies q(0) is finite. 
            # If q(x) > 0 for all x>0, then there are no sign changes for x>0.
            # The problem says "r is the largest real root of q at which q changes sign".
            # If no such root exists, is it valid? 
            # If q(x) > 0 for all x, g(x) > 0 for x>0. g(0)=0.
            # This is a valid configuration? 
            # The prompt implies r exists. Let's assume we need a root.
            # However, if we can make q(x) > 0 for all x, then the "upper bound" logic might allow r=0?
            # "r is the largest real root...". If set is empty, undefined.
            # Let's treat "no positive roots" as invalid for this optimization, 
            # or try to push a root to 0.
            return np.inf, False
        
    r = np.max(pos_roots)
    
    # Check sign change at r
    eps = 1e-6 * (1.0 + abs(r))
    left_val = np.polyval(q, r - eps)
    right_val = np.polyval(q, r + eps)
    
    if left_val * right_val > 0:
        # No sign change, maybe a double root or numerical error.
        # If it's a double root, it doesn't count as "changes sign".
        # We need a simple root.
        # Let's check further out if it's a touch.
        # But for optimization, we prefer simple roots.
        return np.inf, False

    # Check if q(x) > 0 for x > r
    # Sample grid from r to r + 10 (or more)
    x_grid = np.linspace(r, r + 10.0, 200)
    q_vals = np.polyval(q, x_grid)
    if np.any(q_vals < -1e-9):
        return np.inf, False
        
    # Also check behavior at infinity (already ensured by c3 > 0)
    # And check if there are any other sign changes for x > r?
    # The grid check covers x in [r, r+10]. If c3 > 0, it goes to +inf.
    # If it dips below 0, we catch it.
    
    return r * r / (2.0 * _PI), True


def construct():
    t0 = time.time()
    best_val, _ = _compute_value(_SEED[0], _SEED[1], _SEED[2])
    best_c = _SEED.copy()
    
    if not np.isfinite(best_val):
         # Seed might be invalid in our stricter check, try to fix or start random
         best_val = 1.0 # fallback
         best_c = _SEED.copy()

    def _obj(x):
        v, valid = _compute_value(x[0], x[1], x[2])
        if not valid or not np.isfinite(v):
            return 1e10
        return v

    rng = np.random.default_rng(42)
    
    # Strategy: Global search with local refinement.
    # The space is 3D. 
    # We know the seed gives ~0.159.
    # We want to minimize r.
    
    # Scale of search: 
    # c0, c1, c2 can vary. 
    # Let's try a wide box first, then narrow down.
    
    search_ranges = [
        ([-2, 2], [-1, 1], [-0.1, 0.1]),
        ([-5, 5], [-2, 2], [-0.5, 0.5]),
        ([-10, 10], [-5, 5], [-1, 1]),
    ]
    
    for low, high in search_ranges:
        if time.time() - t0 > _DEADLINE - 30:
            break
            
        # Random restarts in this box
        for i in range(500):
            if time.time() - t0 > _DEADLINE - 10:
                break
                
            c0 = rng.uniform(low[0], high[0])
            c1 = rng.uniform(low[1], high[1])
            c2 = rng.uniform(low[2], high[2])
            
            cand = np.array([c0, c1, c2])
            v, valid = _compute_value(c0, c1, c2)
            
            if valid and np.isfinite(v) and v < best_val:
                best_val = v
                best_c = cand.copy()
                
                # Local refinement
                try:
                    res = minimize(_obj, best_c, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.success and np.isfinite(res.fun) and res.fun < best_val:
                        best_val = res.fun
                        best_c = res.x.copy()
                except Exception:
                    pass

    # Final polish around the best found
    if np.isfinite(best_val) and time.time() - t0 < _DEADLINE - 5:
        try:
            res = minimize(_obj, best_c, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Fallback to seed if we failed to find anything better or valid
    if not np.isfinite(best_val):
        return _SEED.copy()
        
    # Check if best is actually better than seed
    seed_val, seed_valid = _compute_value(_SEED[0], _SEED[1], _SEED[2])
    if not seed_valid or best_val < seed_val:
        return best_c
    else:
        return _SEED.copy()

# EVOLVE-BLOCK-END
