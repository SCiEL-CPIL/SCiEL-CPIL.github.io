"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import gamma
import time

def _compute_log_norm(roots):
    """
    Compute log of L2 norm squared of P(x) = prod (x - r_i)^2 with weight e^{-x}.
    MAXIMIZING this norm MINIMIZES the C4 upper bound.
    """
    m = len(roots)
    # Gauss-Laguerre quadrature - use enough nodes for accuracy
    # For degree-24 polynomial squared = degree 48, need at least 25 nodes for exactness
    # But roots can be large, so use more nodes for numerical stability
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x)^2 = prod (x - r_i)^4
    # log(P(x)^2) = 4 * sum log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff = np.maximum(diff, 1e-300)
        log_p_sq += 4.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
    log_integrand = -x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf
    
    sum_val = np.sum(w_quad * np.exp(log_integrand - max_log))
    if sum_val <= 0 or not np.isfinite(sum_val):
        return -np.inf
        
    log_norm_sq = max_log + np.log(sum_val)
    return log_norm_sq

def _c4_bound_from_norm(log_norm_sq):
    """The C4 upper bound is proportional to 1 / ||P||_2^2."""
    if not np.isfinite(log_norm_sq):
        return np.inf
    # The bound is 1 / ||P||_2^2 = exp(-log_norm_sq)
    # But we need to account for the specific Cohn-Goncalves constant
    # The score is the upper bound on C4, so we want to minimize it
    return np.exp(-log_norm_sq)

def _objective_log(params, deadline):
    """Objective: minimize -log(||P||^2) = maximize log(||P||^2)."""
    if time.time() > deadline:
        return 1e10
    roots = np.sort(params)
    if np.any(roots <= 1e-8) or np.any(roots > 500):
        return 1e10
    log_norm = _compute_log_norm(roots)
    if not np.isfinite(log_norm):
        return 1e10
    return -log_norm

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    # Strategy: 
    # 1. Use multiple random starts with local optimization
    # 2. Use a geometric parameterization for global search
    # 3. Final high-precision polish
    
    best_roots = None
    best_score = np.inf
    
    def _evaluate_roots(roots):
        nonlocal best_roots, best_score
        roots = np.sort(roots)
        if len(roots) != 12 or np.any(roots <= 0) or np.any(roots > 500):
            return
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return
        score = np.exp(-log_norm)
        if score < best_score:
            best_score = score
            best_roots = roots.copy()
    
    # Phase 1: Global search using geometric parameterization
    # roots[i] = exp(mu + sigma * i), i = 0..11
    # This ensures positive, sorted roots
    
    def _obj_geometric(params):
        if time.time() > deadline - 20:
            return 1e10
        mu, sigma = params
        if sigma < -0.8 or sigma > 1.5:
            return 1e10
        roots = np.exp(mu + sigma * np.arange(12))
        if np.any(roots <= 1e-8) or np.any(roots > 500):
            return 1e10
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    # Try multiple geometric configurations
    np.random.seed(42)
    
    # Initial guess: roots roughly geometric from ~2 to ~50
    # ratio = (50/2)^(1/11) ≈ 1.41, log ≈ 0.345
    # mu ≈ log(2) ≈ 0.69
    bounds = [(0.0, 4.0), (-0.5, 1.2)]
    
    try:
        de_result = differential_evolution(
            _obj_geometric,
            bounds,
            seed=42,
            maxiter=300,
            tol=1e-10,
            popsize=12,
            init='sobol',
            polish=False,
            updating='deferred'
        )
        if de_result.fun < 1e9:
            mu_opt, sigma_opt = de_result.x
            roots = np.exp(mu_opt + sigma_opt * np.arange(12))
            _evaluate_roots(roots)
    except Exception:
        pass
    
    # Phase 2: Multi-start local optimization in full 12D space
    if time.time() < deadline - 60:
        # Generate diverse starting points
        n_starts = 20
        for s in range(n_starts):
            if time.time() > deadline - 40:
                break
            
            # Different starting strategies
            if s == 0:
                # Geometric from 2 to 40
                start_roots = np.exp(np.linspace(np.log(2), np.log(40), 12))
            elif s == 1:
                # Geometric from 1 to 30
                start_roots = np.exp(np.linspace(np.log(1), np.log(30), 12))
            elif s == 2:
                # Arithmetic from 2 to 48
                start_roots = np.linspace(2, 48, 12)
            elif s == 3:
                # Geometric from 3 to 60
                start_roots = np.exp(np.linspace(np.log(3), np.log(60), 12))
            elif s == 4:
                # Slightly perturbed geometric
                start_roots = np.exp(np.linspace(np.log(2.5), np.log(35), 12))
                start_roots += np.random.normal(0, 0.5, 12)
                start_roots = np.sort(start_roots)
            elif s == 5:
                # Tighter geometric
                start_roots = np.exp(np.linspace(np.log(4), np.log(20), 12))
            elif s == 6:
                # Wider geometric
                start_roots = np.exp(np.linspace(np.log(1.5), np.log(80), 12))
            elif s == 7:
                # Log-uniform random
                start_roots = np.sort(np.exp(np.random.uniform(np.log(1), np.log(100), 12)))
            elif s == 8:
                # Based on Laguerre zeros approximation
                # Zeros of L_m(x) ~ 4m + 4*sqrt(3m)*cos((2k+1)pi/(2m+1))
                # For m=12: 48 + 4*sqrt(36)*cos(...) = 48 + 24*cos(...)
                k = np.arange(12)
                start_roots = 48 + 24 * np.cos((2*k + 1) * np.pi / 25)
                start_roots = np.sort(start_roots)
                start_roots = np.maximum(start_roots, 1.0)
            elif s == 9:
                # Another perturbation
                start_roots = np.exp(np.linspace(np.log(3), np.log(50), 12))
                start_roots += np.random.normal(0, 1.0, 12)
                start_roots = np.sort(start_roots)
                start_roots = np.maximum(start_roots, 0.5)
            else:
                # Random log-uniform
                start_roots = np.sort(np.exp(np.random.uniform(np.log(0.5), np.log(150), 12)))
            
            start_roots = np.sort(start_roots)
            if np.any(start_roots <= 0) or np.any(start_roots > 500):
                continue
            
            try:
                result = minimize(
                    _objective_log,
                    start_roots,
                    args=(deadline - 10,),
                    method='L-BFGS-B',
                    bounds=[(1e-6, 500)] * 12,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if result.fun < 1e9:
                    _evaluate_roots(result.x)
            except Exception:
                pass
    
    # Phase 3: High-precision polish on best result
    if best_roots is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _objective_log,
                best_roots,
                args=(deadline - 2,),
                method='L-BFGS-B',
                bounds=[(1e-6, 500)] * 12,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if result.fun < 1e9:
                _evaluate_roots(result.x)
        except Exception:
            pass
    
    # Final fallback
    if best_roots is None:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
    
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        best_roots = np.array([4.0 * i for i in range(1, 13)])
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
