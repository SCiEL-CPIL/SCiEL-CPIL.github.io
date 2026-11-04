import numpy as np
from scipy.special import eval_hermite, roots_hermite
from scipy.optimize import differential_evolution, minimize
import time

# Precompute Gauss-Hermite nodes and weights
# We need exact integration of P(x)^4 which has degree 48.
# N=30 points is exact for degree up to 59.
_N_GH = 30
_gh_nodes, _gh_weights = roots_hermite(_N_GH)

# Transformation: integral P(x)^k exp(-pi x^2) dx = 1/sqrt(pi) * integral P(u/sqrt(pi))^k exp(-u^2) du
# Let x = u / sqrt(pi)
# H_n(x) = H_n(u / sqrt(pi))
_sqrt_pi = np.sqrt(np.pi)
_x_transformed = _gh_nodes / _sqrt_pi

# Precompute Hermite polynomials at these points
# We need H_0, H_4, H_8, H_12
_H0 = np.ones_like(_x_transformed)
_H4 = eval_hermite(4, _x_transformed)
_H8 = eval_hermite(8, _x_transformed)
_H12 = eval_hermite(12, _x_transformed)

# Basis matrix for fast evaluation
# Shape: (N_GH, 4)
_basis = np.column_stack((_H0, _H4, _H8, _H12))

# Weights for the transformed integral
# integral = (1/sqrt(pi)) * sum( w_i * P(u_i/sqrt(pi))^k )
_w_eff = _gh_weights / _sqrt_pi

def _objective(coeffs):
    """
    Compute the objective function A(g) * A(g_hat).
    Since g_hat has the same coefficients (for n=0,4,8,12), A(g) = A(g_hat).
    Objective = A(g)^2.
    A(g) = int |g|^4 / (int |g|^2)^2
    """
    a0, a4, a8 = coeffs
    
    # Calculate a12 such that P(0) = 0
    # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640
    # a0 - 12 a4 + 1680 a8 - 332640 a12 = 0
    a12 = (a0 - 12.0 * a4 + 1680.0 * a8) / 332640.0
    
    c = np.array([a0, a4, a8, a12])
    
    # Evaluate P at all GH nodes
    P = _basis @ c
    
    # Compute integrals
    # norm2_sq = int P^2 exp(-pi x^2) dx
    norm2_sq = np.dot(_w_eff, P**2)
    
    # norm4_sq = int P^4 exp(-pi x^2) dx
    norm4_sq = np.dot(_w_eff, P**4)
    
    if norm2_sq < 1e-15:
        return 1e10
        
    A_g = norm4_sq / (norm2_sq * norm2_sq)
    
    # Since A_g_hat = A_g, objective is A_g^2
    return A_g * A_g

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit
    
    # Seed values
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    # Evaluate seed
    seed_score = _objective(x0)
    
    best_result = x0.copy()
    best_score = seed_score
    
    # Bounds for Differential Evolution
    # Based on seed and typical scales
    bounds = [(-10.0, 10.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    # Global Search with Differential Evolution
    # popsize=15, tol=1e-8, maxiter=100
    # This might be slow, so we monitor time.
    try:
        de_res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-8, 
            popsize=15,
            polish=False
        )
        
        if time.time() < deadline - 20:
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_result = de_res.x.copy()
                
    except Exception:
        pass
    
    # Local Refinement with L-BFGS-B
    # Start from the best result found so far
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective, 
                best_result, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    # Additional local refinement from seed if DE didn't improve much
    if time.time() < deadline - 5 and not np.allclose(best_result, x0, rtol=1e-5):
         # If we moved away, maybe we should also check if a local min near seed is better? 
         # Actually, DE should have covered it. 
         # Let's just do one more polish on the current best.
         pass
         
    # Final polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _objective, 
                best_result, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_result)):
        return x0
        
    return best_result
