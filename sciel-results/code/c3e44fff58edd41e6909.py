"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_bound(a0, a4, a8):
    """
    Compute the Heisenberg uncertainty product sigma_x * sigma_p for the given Hermite coefficients.
    Pre-computed matrices for speed.
    """
    # H values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 4790016.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Normalization constants
    N_0 = np.sqrt(np.pi)
    N_4 = np.sqrt(384 * np.pi)
    N_8 = np.sqrt(10321920 * np.pi)
    N_12 = np.sqrt(4096 * 479001600 * np.pi)
    
    c_0 = a0 * N_0
    c_4 = a4 * N_4
    c_8 = a8 * N_8
    c_12 = a12 * N_12
    
    c = np.array([c_0, c_4, c_8, c_12])
    
    norm_sq = c[0]*c[0] + c[1]*c[1] + c[2]*c[2] + c[3]*c[3]
    if norm_sq < 1e-12:
        return 1e10
        
    # Pre-computed matrix elements
    # n_vals = [0, 4, 8, 12]
    # M_x and M_p differ only in sign of off-diagonals for |n-m|=2
    
    # Diagonals
    d0 = 0.5
    d1 = 4.5
    d2 = 8.5
    d3 = 12.5
    
    # Off-diagonals for |n-m|=2
    # (0,2): n_low=0 -> val = 0
    # (1,3): n_low=4 -> val = 0.5 * sqrt(4*5) = 0.5 * sqrt(20) = sqrt(5)
    # Others are 0
    
    v13 = np.sqrt(5.0)
    
    # x2_expect = (c0^2*d0 + c1^2*d1 + c2^2*d2 + c3^2*d3 + 2*v13*c1*c3) / norm_sq
    # p2_expect = (c0^2*d0 + c1^2*d1 + c2^2*d2 + c3^2*d3 - 2*v13*c1*c3) / norm_sq
    
    c0_sq = c[0]*c[0]
    c1_sq = c[1]*c[1]
    c2_sq = c[2]*c[2]
    c3_sq = c[3]*c[3]
    c1_c3 = c[1]*c[3]
    
    base = c0_sq*d0 + c1_sq*d1 + c2_sq*d2 + c3_sq*d3
    off = 2.0 * v13 * c1_c3
    
    x2_expect = (base + off) / norm_sq
    p2_expect = (base - off) / norm_sq
    
    if x2_expect < 0 or p2_expect < 0:
        return 1e10
    prod = np.sqrt(x2_expect * p2_expect)
    
    return prod

def construct():
    """
    Find optimal Hermite coefficients to minimize the upper bound C4 (Heisenberg product).
    """
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_bound = _compute_bound(*x0)
    best_x = x0.copy()
    
    # Analytical reduction:
    # Let u = c1 (coeff of H4), v = c3 (coeff of H12)
    # x2 = (base + 2*sqrt(5)*u*v) / norm
    # p2 = (base - 2*sqrt(5)*u*v) / norm
    # prod = sqrt(x2*p2) = sqrt((base^2 - (2*sqrt(5)*u*v)^2) / norm^2)
    # = sqrt(base^2 - 20*u^2*v^2) / norm
    # where base = 0.5*c0^2 + 4.5*c1^2 + 8.5*c2^2 + 12.5*c3^2
    # norm = c0^2 + c1^2 + c2^2 + c3^2
    
    # We can reparameterize to directly minimize this expression.
    # But let's stick with the 3-parameter space (a0, a4, a8) as required.
    
    # Multi-start optimization with more starts and wider exploration
    np.random.seed(42)
    n_starts = 50
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        if i == 0:
            x_start = x0.copy()
        else:
            # Random perturbation with varying scales
            scales = np.array([1.0, 0.04, 0.0003]) # Approximate scales of x0
            exploration_factor = 0.1 + 0.9 * (i / n_starts)
            x_start = x0 + np.random.randn(3) * scales * exploration_factor * 5.0
        
        try:
            result = minimize(
                _compute_bound,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            
            if np.isfinite(result.fun) and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
            
    # Final high-precision polish pass
    if time.time() < deadline - 2:
        try:
            result_final = minimize(
                _compute_bound,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if np.isfinite(result_final.fun) and result_final.fun < best_bound:
                best_bound = result_final.fun
                best_x = result_final.x.copy()
        except Exception:
            pass
            
    return best_x
# EVOLVE-BLOCK-END
