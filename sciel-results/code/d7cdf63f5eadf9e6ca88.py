"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature
    # n_quad=200 is usually sufficient for high accuracy
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) = prod (x - x_i)^2
    # To avoid underflow/overflow issues in the product, 
    # we can compute log_P and then exp, but we need P and P^2.
    # Given n=12, values are manageable in float64 for x in [0, 50].
    
    # Vectorized computation of P(x_quad)
    # P(x) = prod_{i=1}^{12} (x - x_i)^2
    # We can compute this as (prod_{i=1}^{12} (x - x_i))^2
    # Let Q(x) = prod (x - x_i). Then P(x) = Q(x)^2.
    
    # Calculate Q(x_quad)
    # Q(x) is a polynomial of degree 12.
    # We can evaluate it using np.poly1d or just loops for speed/clarity.
    # For n=12, a loop over 12 roots for each of 200 quadrature points is 2400 ops, very fast.
    
    Q_vals = np.ones(n_quad)
    for r in roots:
        Q_vals *= (x_quad - r)
        
    P_vals = Q_vals * Q_vals
    P2_vals = P_vals * P_vals
    
    # Integrate
    int_P = np.sum(w_quad * P_vals)
    int_P2 = np.sum(w_quad * P2_vals)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2) or np.isnan(int_P) or np.isnan(int_P2):
        return np.inf
        
    c4 = int_P2 / (int_P * int_P)
    return c4

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 600
    
    # Initial guess: Standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    # Bounds for optimization
    # Laguerre roots for n=12 are in [0, ~25]. 
    # Allowing a wider range helps global search.
    lower_bound = 1e-4
    upper_bound = 50.0
    bounds = [(lower_bound, upper_bound)] * 12
    
    # Global Optimization using Differential Evolution
    # DE is robust for non-convex problems.
    # We limit maxiter to ensure we finish in time.
    # popsize=15 is a good default.
    
    try:
        # Check time before starting DE
        if time.time() < deadline - 300:
            de_result = differential_evolution(
                _compute_c4_bound,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=False, # We will polish separately
                init='sobol'
            )
            
            if de_result.success or de_result.fun < best_score:
                candidate = de_result.x
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
    except Exception as e:
        pass
    
    # Local Refinement
    # Use L-BFGS-B or Powell to refine the best candidate found
    if time.time() < deadline - 60:
        try:
            # L-BFGS-B is efficient for smooth functions
            # Powell is derivative-free and can be more robust if gradients are noisy
            # Let's try L-BFGS-B first
            
            # To help L-BFGS-B, we might want to sort the roots?
            # No, L-BFGS-B works on the vector. Permutation symmetry can cause issues with gradients.
            # However, since the function is symmetric, any permutation is a minimum.
            # The optimizer will likely converge to one of them.
            
            result = minimize(
                _compute_c4_bound,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_score:
                candidate = result.x
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
                        
        except Exception:
            pass
            
    # One more pass with Powell if time permits, as it handles non-smoothness better
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _compute_c4_bound,
                best_roots,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'xtol': 1e-12}
            )
            
            if result.fun < best_score:
                candidate = result.x
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Ensure final output is sorted and positive
    best_roots = np.sort(best_roots)
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
