"""Rewritten program for the refined Cohn-Goncalves Laguerre construction.
Uses multi-start L-BFGS-B in log-space with efficient cost evaluation."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    m = 12
    n = 1

    start = time.time()
    DEADLINE = 630.0
    MARGIN = 60.0  # Larger margin to be safe

    def ok():
        return (time.time() - start) < (DEADLINE - MARGIN)

    # Precompute Laguerre basis coefficients and inverse transform
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1) ** j) * math.comb(k + 1, j + 1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
        
    w = np.arange(13) + 1

    def compute_norm_sq(roots):
        # Use np.poly for coefficient extraction
        # Roots are assumed positive and reasonably sized
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c ** 2 * w)
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    def cost_with_grad(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros_like(log_roots)
        
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c ** 2 * w)
        if norm_sq <= 1e-10:
            return 1e10, np.zeros_like(log_roots)
        
        cost = 1.0 / norm_sq
        grad = np.zeros(12)
        
        # Compute gradient analytically
        for i in range(12):
            other_roots = np.delete(roots, i)
            coeffs_other_desc = np.poly(other_roots)
            p_other = coeffs_other_desc[::-1]
            # dp/dr_i = - product_{j!=i} (x - r_j) = -p_other
            dp_dr_i = -p_other
            dc_dr_i = A_T_inv @ dp_dr_i
            dnorm_sq_dr_i = 2.0 * np.sum(c * w * dc_dr_i)
            dcost_dr_i = -dnorm_sq_dr_i / (norm_sq * norm_sq)
            grad[i] = dcost_dr_i * roots[i]
            
        return cost, grad

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Generate initial points
    # Heuristic: Laguerre roots for large m are roughly ~ i^2 or similar
    # For m=12, roots are in the range [0, ~100]
    initial_points = []
    
    # Start 1: Geometric progression
    r1 = 2.0 * (1.5 ** np.arange(12))
    initial_points.append(np.log(r1))
    
    # Start 2: Arithmetic progression
    r2 = np.linspace(1, 50, 12)
    initial_points.append(np.log(r2))
    
    # Start 3: Random uniform in log space
    np.random.seed(42)
    for _ in range(10):
        lr = np.random.uniform(-1.0, 5.0, 12)
        initial_points.append(lr)
        
    # Phase 1: Multi-start L-BFGS-B
    for i, init_lr in enumerate(initial_points):
        if not ok():
            break
        try:
            res = minimize(cost_with_grad, init_lr, jac=True, method='L-BFGS-B',
                           bounds=[(-4.0, 10.0)] * 12,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 2: Differential Evolution (limited)
    if ok():
        try:
            res = differential_evolution(cost_func, [(-4.0, 10.0)] * 12,
                                         seed=42, maxiter=50, popsize=15,
                                         tol=1e-8, polish=True)
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 3: Final Polish
    if best_log_roots is not None and ok():
        try:
            res = minimize(cost_with_grad, best_log_roots, jac=True, method='L-BFGS-B',
                           bounds=[(-6.0, 12.0)] * 12,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    if best_roots is not None:
        # Ensure positivity
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)

    # Fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
