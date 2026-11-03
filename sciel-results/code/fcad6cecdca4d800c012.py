import numpy as np
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
import time

# Precompute Gauss-Hermite nodes and weights
# N=20 is sufficient for high accuracy (integrates up to degree 39)
N_QUAD = 20
_x_nodes, _w_weights = hermgauss(N_QUAD)

# Precompute Hermite polynomial values at nodes
_H0 = np.ones(N_QUAD)
_H4 = eval_hermite(4, _x_nodes)
_H8 = eval_hermite(8, _x_nodes)
_H12 = eval_hermite(12, _x_nodes)

# Values at x=0 for constraint (Physicist's Hermite)
_h0_0 = 1.0
_h4_0 = 3.0
_h8_0 = 105.0
_h12_0 = 945.0

# Precompute basis vectors for P(x) satisfying P(0)=0
# P(x) = c0*A + c4*B + c8*C
# A = H0 - H12/945
# B = H4 - 3*H12/945
# C = H8 - 105*H12/945
_A = _H0 - _H12 / _h12_0
_B = _H4 - _h4_0 * _H12 / _h12_0
_C = _H8 - _h8_0 * _H12 / _h12_0

_x_sq = _x_nodes**2

def _compute_bound(c):
    """Compute the upper bound C4 for given coefficients c = [c0, c4, c8]."""
    # Normalize c to unit length to handle scale invariance
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return 1e10
    c = c / norm
    
    c0, c4, c8 = c
    
    # Compute P(x) at nodes
    P = c0 * _A + c4 * _B + c8 * _C
    
    P_sq = P * P
    norm_sq = np.sum(_w_weights * P_sq)
    
    if norm_sq < 1e-15:
        return 1e10
        
    second_moment = np.sum(_w_weights * _x_sq * P_sq) / norm_sq
    C4 = second_moment ** 2
    return C4

def construct():
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    best_c = None
    best_val = float('inf')
    
    # Seed point from previous best
    seed_c = np.array([-1.13, 0.04, 0.00030864])
    seed_val = _compute_bound(seed_c)
    best_val = seed_val
    best_c = seed_c.copy()
    
    # Multi-start local optimization
    # We generate random directions on the sphere and polish them.
    # This is efficient for 3D smooth landscapes.
    
    rng = np.random.default_rng(42)
    
    num_starts = 200
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Random point on the sphere
        c_rand = rng.normal(size=3)
        c_rand = c_rand / np.linalg.norm(c_rand)
        
        # Try Nelder-Mead
        try:
            res = minimize(
                _compute_bound, 
                c_rand, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
        
        # Occasionally try L-BFGS-B as well, though NM is often better for non-smooth-ish (due to normalization)
        if i % 10 == 0:
            try:
                res_l = minimize(
                    _compute_bound, 
                    c_rand, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                if res_l.fun < best_val:
                    best_val = res_l.fun
                    best_c = res_l.x.copy()
            except Exception:
                pass

    # Final polish from the best found
    if best_c is not None:
        try:
            res_final = minimize(
                _compute_bound, 
                best_c, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_c = res_final.x.copy()
        except Exception:
            pass

    if best_c is None:
        return seed_c

    return np.array(best_c)
