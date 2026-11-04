"""Refined Cohn-Goncalves Laguerre double-root construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _compute_norm_log_and_grad(roots, n_quad=800):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)^2
    (double roots) with weight e^{-x}, and its gradient w.r.t. roots.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound.
    
    P(x) = prod_i (x - r_i)^2
    |P(x)|^2 = prod_i (x - r_i)^4
    log |P(x)|^2 = 4 * sum_i log|x - r_i|
    """
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log |P(x)|^2 = 4 * sum_i log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 4.0 * np.log(diff)
    
    # Integrand: e^{-x} * |P(x)|^2
    log_integrand = -x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf, np.zeros(m)
    
    exp_shifted = np.exp(log_integrand - max_log)
    weighted = w_quad * exp_shifted
    sum_val = np.sum(weighted)
    
    if sum_val <= 0:
        return -np.inf, np.zeros(m)
        
    log_norm_sq = max_log + np.log(sum_val)
    
    # Gradient: d/d(r_j) log_norm_sq = 4 * sum_i weighted_i * (1/(x_i - r_j))
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        diff[np.abs(diff) < 1e-300] = 1e-300
        grad[j] = 4.0 * np.sum(weighted / diff)
    
    return log_norm_sq, grad

def _dyson_gas_init(m=12, repulsion=4.0):
    """
    Initialize root positions using Dyson gas equilibrium with repulsion exponent
    (matching double roots -> repulsion ~ 4) and linear confining potential.
    """
    roots = np.linspace(2.0, 50.0, m)
    
    for iteration in range(2000):
        forces = np.zeros(m)
        for i in range(m):
            forces[i] = roots[i]
            for j in range(m):
                if i != j:
                    forces[i] -= repulsion / (roots[i] - roots[j])
        
        max_force = np.max(np.abs(forces))
        step_size = min(0.05, 1.0 / (1.0 + max_force))
        for i in range(m):
            roots[i] += step_size * forces[i]
            roots[i] = max(roots[i], 0.5)
        
        if max_force < 1e-14:
            break
    
    return np.sort(roots)

def _objective_and_grad(log_params, n_quad=800):
    """Objective and gradient in log-space for maximizing the norm."""
    roots = np.exp(log_params)
    roots = np.sort(roots)
    
    if any(r <= 1e-6 or r > 1e6 for r in roots):
        return 1e10, np.zeros(len(log_params))
    
    log_norm, grad = _compute_norm_log_and_grad(roots, n_quad)
    if not np.isfinite(log_norm):
        return 1e10, np.zeros(len(log_params))
    
    # Chain rule: d/d(log_r_j) = r_j * d/d(r_j)
    grad_log = grad * roots
    return -log_norm, -grad_log

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial candidates
    try:
        dyson_roots = _dyson_gas_init(12, repulsion=4.0)
    except Exception:
        dyson_roots = np.array([4.0 * i for i in range(1, 13)])
    
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Evaluate both starting points
    best_roots = dyson_roots.copy()
    best_log_norm, _ = _compute_norm_log_and_grad(best_roots)
    
    seed_log_norm, _ = _compute_norm_log_and_grad(seed_roots)
    if seed_log_norm > best_log_norm:
        best_roots = seed_roots.copy()
        best_log_norm = seed_log_norm
    
    # Strategy 1: L-BFGS-B in log-space from best initial point
    try:
        log_init = np.log(best_roots)
        result = minimize(_objective_and_grad, log_init, jac=True,
                         method='L-BFGS-B',
                         bounds=[(np.log(0.5), np.log(300.0))] * 12,
                         options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if result.success:
            candidate_roots = np.sort(np.exp(result.x))
            cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
            if cand_log_norm > best_log_norm:
                best_roots = candidate_roots
                best_log_norm = cand_log_norm
    except Exception:
        pass
    
    # Strategy 2: Differential Evolution for global search if time permits
    if time.time() - start_time < deadline - 200:
        try:
            def _de_objective(log_params):
                obj, _ = _objective_and_grad(log_params)
                return obj
            
            bounds_de = [(np.log(1.0), np.log(200.0))] * 12
            result_de = differential_evolution(
                _de_objective, bounds_de,
                seed=42, maxiter=50, tol=1e-10,
                polish=True, init='sobol',
                popsize=15, mutation=(0.3, 0.8), recombination=0.8
            )
            
            if result_de.success:
                candidate_roots = np.sort(np.exp(result_de.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
        except Exception:
            pass
    
    # Strategy 3: Polish with tighter tolerances
    if time.time() - start_time < deadline - 60:
        try:
            log_best = np.log(best_roots)
            result3 = minimize(_objective_and_grad, log_best, jac=True,
                             method='L-BFGS-B',
                             bounds=[(np.log(0.5), np.log(300.0))] * 12,
                             options={'maxiter': 2000, 'maxfun': 100000, 'ftol': 1e-18, 'gtol': 1e-16})
            
            if result3.success:
                candidate_roots = np.sort(np.exp(result3.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
        except Exception:
            pass
    
    # Strategy 4: Nelder-Mead for local refinement if time permits
    if time.time() - start_time < deadline - 30:
        try:
            def _nm_objective(log_params):
                obj, _ = _objective_and_grad(log_params, n_quad=500)
                return obj
            
            result4 = minimize(_nm_objective, np.log(best_roots), method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if result4.success:
                candidate_roots = np.sort(np.exp(result4.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
        except Exception:
            pass
    
    # Final verification: Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = seed_roots.copy()
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
