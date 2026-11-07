"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy import special
import sys

# Precompute Hermite polynomial coefficients
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def compute_q_coeffs(c0, c1, c2):
    """Return coefficients of q(y) = g(x)/x^2 as polynomial in y=x^2.
    q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    Returns [a0, a1, a2, a3, a4, a5] (ascending order)."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    a0 = 0.0
    a1 = c1 * (-48) + c2 * (-1344) + c3 * (-921600)
    a2 = c1 * 16 + c2 * 2688 + c3 * 1424640
    a3 = c2 * (-1344) + c3 * (-1003520)
    a4 = c2 * 256 + c3 * 332160
    a5 = c3 * (-46080)
    a6 = c3 * 4096
    
    return np.array([a0, a1, a2, a3, a4, a5, a6])

def find_largest_sign_change_root(q_coeffs):
    """Find the largest positive real root of q(y) at which q changes sign.
    q_coeffs is [a0, a1, ..., a6] ascending."""
    # Use numpy to find roots
    # np.roots expects descending order
    desc = q_coeffs[::-1]
    
    # Remove leading zeros
    nonzero = np.nonzero(desc)[0]
    if len(nonzero) == 0:
        return None
    desc = desc[nonzero[0]:]
    
    if len(desc) <= 1:
        return None
    
    roots = np.roots(desc)
    
    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    # Check sign change at each root
    def q_eval(y):
        return np.polyval(desc, y)
    
    best_root = None
    for r in sorted(real_roots, reverse=True):
        # Check sign change
        eps = 1e-8 * max(1.0, abs(r))
        q_below = q_eval(r - eps)
        q_above = q_eval(r + eps)
        if q_below * q_above < 0:
            best_root = r
            break
    
    return best_root

def evaluate_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients."""
    q_coeffs = compute_q_coeffs(c0, c1, c2)
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    if c3 > 0:
        # g tends to +infinity, so we replace g by -g
        q_coeffs = -q_coeffs
    
    r = find_largest_sign_change_root(q_coeffs)
    
    if r is None:
        return float('inf')
    
    return r**2 / (2 * np.pi)

def objective_2d(x):
    """Objective for 2D optimization: x = [c1, c2], c0 fixed at 1."""
    c0, c1, c2 = 1.0, x[0], x[1]
    try:
        val = evaluate_value(c0, c1, c2)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # safety margin before 210s kill
    
    # Scale invariance: fix c0 = 1, optimize (c1, c2)
    # Seed: c0=-1.13, c1=0.04, c2=1/3240
    # Normalize: divide by c0 -> c0'=1, c1'=0.04/(-1.13), c2'=(1/3240)/(-1.13)
    seed_c1 = 0.04 / (-1.13)
    seed_c2 = (1.0/3240.0) / (-1.13)
    
    best_val = evaluate_value(1.0, seed_c1, seed_c2)
    best_x = np.array([seed_c1, seed_c2])
    
    # Phase 1: Coarse grid search in 2D
    elapsed = time.time() - start_time
    if elapsed < deadline - 80:
        # Grid search with adaptive resolution
        n_grid = 40
        c1_range = np.linspace(-5, 5, n_grid)
        c2_range = np.linspace(-5, 5, n_grid)
        
        for c1 in c1_range:
            for c2 in c2_range:
                if time.time() > deadline - 80:
                    break
                val = objective_2d([c1, c2])
                if val < best_val:
                    best_val = val
                    best_x = np.array([c1, c2])
    
    # Phase 2: Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        try:
            res = minimize(objective_2d, best_x, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except:
            pass
    
    # Phase 3: Try multiple random starting points with local refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        rng = np.random.RandomState(42)
        n_random = 100
        for i in range(n_random):
            if time.time() > deadline - 20:
                break
            x_try = rng.uniform(-10, 10, size=2)
            val = objective_2d(x_try)
            if val < best_val:
                best_val = val
                best_x = x_try.copy()
                # Quick local refinement
                try:
                    res2 = minimize(objective_2d, x_try, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                                           'adaptive': True})
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_x = res2.x
                except:
                    pass
    
    # Phase 4: Final polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            res3 = minimize(objective_2d, best_x, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18,
                                   'adaptive': True})
            if res3.fun < best_val:
                best_val = res3.fun
                best_x = res3.x
        except:
            pass
    
    # Convert back to 3D: c0=1, c1, c2
    result = np.array([1.0, best_x[0], best_x[1]])
    
    return result
# EVOLVE-BLOCK-END
