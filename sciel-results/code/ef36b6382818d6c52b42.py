"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize

# Precompute Gauss-Hermite nodes and weights
# We need to integrate up to degree 26 (x^2 * P(x)^2 where P is degree 12)
# N=20 is safe (integrates exactly up to degree 39)
N_QUAD = 20
_x_nodes, _w_weights = hermgauss(N_QUAD)
_x_sq = _x_nodes**2

def _get_c12(c0, c4, c8):
    # H_0(0) = 1
    # H_4(0) = -3
    # H_8(0) = 105
    # H_12(0) = -945
    # c0*1 + c4*(-3) + c8*105 + c12*(-945) = 0
    return (c0 - 3*c4 + 105*c8) / 945.0

def _compute_bound(c0, c4, c8):
    c12 = _get_c12(c0, c4, c8)
    
    # Evaluate P(x) at nodes
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # H_n(x) for physicist's Hermite polynomials
    # eval_hermite(n, x) gives H_n(x)
    
    h0 = np.ones_like(_x_nodes)
    h4 = eval_hermite(4, _x_nodes)
    h8 = eval_hermite(8, _x_nodes)
    h12 = eval_hermite(12, _x_nodes)
    
    p_vals = c0*h0 + c4*h4 + c8*h8 + c12*h12
    
    p_sq = p_vals**2
    
    num = np.sum(_w_weights * _x_sq * p_sq)
    den = np.sum(_w_weights * p_sq)
    
    if den < 1e-15:
        return 1e10
        
    # The uncertainty product is M2_x * M2_xi.
    # For these functions, M2_x == M2_xi.
    # So product is (M2_x)^2.
    # Minimizing the product is equivalent to minimizing M2_x.
    # However, the "value" might be defined as the product itself.
    # Let's return the product to be safe, or just M2_x?
    # The prompt says "upper bound on A(f)*A(Fourier(f))".
    # This suggests the product.
    m2 = num / den
    return m2 * m2

def construct():
    t_start = time.time()
    deadline = t_start + 200.0  # 10s safety margin
    
    # Baseline coefficients
    c_base = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    def objective(c):
        return _compute_bound(c[0], c[1], c[2])
    
    best_c = c_base.copy()
    best_val = objective(best_c)
    
    # Optimization
    # Nelder-Mead is robust for small dimensional problems without gradients
    try:
        res = minimize(
            objective, 
            best_c, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    except Exception:
        pass
    
    # Check time and maybe try a second run or different method if time allows
    if time.time() < deadline - 5.0:
        try:
            # Powell method is another derivative-free option
            res2 = minimize(
                objective, 
                best_c, 
                method='Powell', 
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x
        except Exception:
            pass
            
    return np.array(best_c)

# EVOLVE-BLOCK-END
