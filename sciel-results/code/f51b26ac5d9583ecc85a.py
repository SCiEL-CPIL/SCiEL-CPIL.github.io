import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

# Pre-compute Gauss-Hermite nodes and weights
# N=100 is overkill for degree 48 polynomial (25 is exact), but 100 is safe and fast.
_N_GH = 100
_t_nodes, _w_weights = np.polynomial.hermite.hermgauss(_N_GH)

# Scaling factors for x = t / sqrt(2*pi) and x = t / sqrt(4*pi)
# Integral I2 = int P(x)^2 exp(-2*pi*x^2) dx
# Let u = sqrt(2*pi)*x => dx = du/sqrt(2*pi)
# I2 = 1/sqrt(2*pi) * int P(u/sqrt(2*pi))^2 exp(-u^2) du
# Approx: 1/sqrt(2*pi) * sum w_i P(t_i/sqrt(2*pi))^2

# Integral I4 = int P(x)^4 exp(-4*pi*x^2) dx
# Let v = sqrt(4*pi)*x => dx = dv/sqrt(4*pi)
# I4 = 1/sqrt(4*pi) * int P(v/sqrt(4*pi))^4 exp(-v^2) du
# Approx: 1/sqrt(4*pi) * sum w_i P(t_i/sqrt(4*pi))^4

_c2 = 1.0 / np.sqrt(2.0 * np.pi)
_c4 = 1.0 / np.sqrt(4.0 * np.pi)

_x2_nodes = _t_nodes * _c2
_x4_nodes = _t_nodes * _c4

_w2_nodes = _w_weights * _c2
_w4_nodes = _w_weights * _c4

# Pre-compute Hermite polynomials at these specific nodes
# H0, H4, H8, H12
_H0_2 = np.ones_like(_x2_nodes)
_H4_2 = eval_hermite(4, _x2_nodes)
_H8_2 = eval_hermite(8, _x2_nodes)
_H12_2 = eval_hermite(12, _x2_nodes)

_H0_4 = np.ones_like(_x4_nodes)
_H4_4 = eval_hermite(4, _x4_nodes)
_H8_4 = eval_hermite(8, _x4_nodes)
_H12_4 = eval_hermite(12, _x4_nodes)

# Weights for the sums
# Note: We multiply P^k by w_i * scale.
# So we can just compute P at nodes and dot with (w_i * scale).

def _compute_objective_fast(coeffs):
    a0, a4, a8 = coeffs
    
    # Calculate a12 such that P(0) = 0
    # P(0) = a0*1 + a4*(-12) + a8*(1680) + a12*(-332640) = 0
    # a12 = (a0 - 12*a4 + 1680*a8) / 332640
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 332640.0
    
    # Compute P(x) at x2_nodes and x4_nodes
    # P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    # For I2 (norm2_sq and norm4_sq on direct side? No, norm2 is I2, norm4 is I4)
    # Wait, the objective uses:
    # norm2_sq = int |g|^2 = I2(P)
    # norm4_sq = int |g|^4 = I4(P)
    # norm2_sq_hat = int |g_hat|^2 = I2(P_minus)
    # norm4_sq_hat = int |g_hat|^4 = I4(P_minus)
    
    # Direct side (P)
    P2 = a0*_H0_2 + a4*_H4_2 + a8*_H8_2 + a12*_H12_2
    P4 = a0*_H0_4 + a4*_H4_4 + a8*_H8_4 + a12*_H12_4
    
    # Fourier side (P_minus)
    Pm2 = a0*_H0_2 + a4*_H4_2 + a8*_H8_2 - a12*_H12_2
    Pm4 = a0*_H0_4 + a4*_H4_4 + a8*_H8_4 - a12*_H12_4
    
    # Integrate
    # I2(P) = sum(P2^2 * _w2_nodes)
    # I4(P) = sum(P4^4 * _w4_nodes)
    
    norm2_sq = np.sum(P2**2 * _w2_nodes)
    norm4_sq = np.sum(P4**4 * _w4_nodes)
    
    norm2_sq_hat = np.sum(Pm2**2 * _w2_nodes)
    norm4_sq_hat = np.sum(Pm4**4 * _w4_nodes)
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
        
    obj = (norm4_sq / (norm2_sq * norm2_sq)) * (norm4_sq_hat / (norm2_sq_hat * norm2_sq_hat))
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = x0
    best_score = _compute_objective_fast(x0)
    
    # 1. Local Optimization from Seed and Perturbations
    np.random.seed(42)
    
    starts = [x0]
    for i in range(10):
        starts.append(x0 + np.random.randn(3) * np.array([0.1, 0.01, 0.0001]))
        
    for s in starts:
        if time.time() > deadline - 60: # Leave 60s for DE
            break
        try:
            res = minimize(_compute_objective_fast, s, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # 2. Differential Evolution for Global Search
    # Bounds based on seed magnitudes
    # a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
    
    if time.time() < deadline - 30:
        try:
            # DE is stochastic, but we want deterministic behavior for reproducibility if possible?
            # Seed is fixed.
            de_res = differential_evolution(_compute_objective_fast, bounds, 
                                            seed=42, maxiter=50, tol=1e-8, 
                                            polish=True, init='sobol')
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_result = de_res.x
        except Exception:
            pass
            
    # 3. Final Polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_compute_objective_fast, best_result, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    return best_result
