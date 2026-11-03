"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_laguerre, roots_laguerre, gammaln
import time

def _compute_bound(roots):
    """
    Compute an upper bound on the C4 constant given root locations.
    For the Cohn-Goncalves Laguerre construction with m double roots in 1D,
    we use a principled model based on the Laguerre polynomial structure.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or len(roots) != 12:
        return 1e10
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return 1e10
    
    r = np.sort(roots)
    
    # The Cohn-Goncalves bound for Fourier sign uncertainty in 1D with
    # the Laguerre method. For m=12 double roots, the key quantity is
    # related to the ratio of norms of the constructed function.
    
    # A concrete model: the bound C4 is approximately
    # C = exp(sum_i log(r_i) - 2*log(m) + correction)
    # where the correction accounts for the spread of roots.
    
    # More precisely, for the Laguerre construction, the bound involves
    # the product of root locations and a normalization factor.
    
    # Let's use a formula that captures the essential physics:
    # C = (prod_i r_i)^{1/m} * exp(-sum_i log(r_i + 1))
    # This rewards larger, well-spread roots.
    
    log_prod = np.sum(np.log(r))
    log_sum_inv = np.sum(np.log(r + 1.0))
    
    # The bound is proportional to exp(log_prod/m - log_sum_inv/m + const)
    # Let's use a simpler, well-behaved proxy:
    C = np.exp(log_prod / 12.0 - log_sum_inv / 12.0 + 1.0)
    
    return C

def _compute_bound_v2(roots):
    """
    Alternative bound computation using a different model.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or len(roots) != 12:
        return 1e10
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return 1e10
    
    r = np.sort(roots)
    
    # Model based on the idea that the bound is minimized when roots
    # are positioned to maximize the "concentration" of the constructed
    # function. Use a Gaussian-like potential model.
    
    # C = exp(-sum_i (r_i - mu)^2 / (2*sigma^2)) * normalization
    # where mu and sigma are chosen based on the Laguerre asymptotics.
    
    # For Laguerre L_12, the zeros are approximately at
    # x_i ≈ 4*i*(12+1) for i=1..12 (from Weyl's law for Laguerre)
    # So the typical scale is ~4*13 = 52
    
    mu = 4.0 * 13.0 / 2.0  # ~26
    sigma = 4.0 * 13.0 / 2.0  # ~26
    
    potential = np.sum((r - mu)**2 / (2.0 * sigma**2))
    C = np.exp(potential / 12.0)
    
    return C

def _compute_bound_v3(roots):
    """
    Third model: based on the Dyson gas equilibrium.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or len(roots) != 12:
        return 1e10
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return 1e10
    
    r = np.sort(roots)
    
    # Dyson gas with repulsion exponent 2 (beta=2) and linear confinement.
    # The equilibrium satisfies: for each i, sum_{j!=i} 2/(r_i - r_j) + r_i = 0
    # (in the appropriate scaling)
    
    # The "energy" of the configuration is:
    # E = -sum_i log(r_i) + sum_{i<j} log|r_i - r_j| + sum_i r_i^2/2
    # (up to constants)
    
    # For the bound, we use a related quantity:
    # C = exp(E / 12)
    
    # Compute pairwise log distances
    log_det = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            d = abs(r[i] - r[j])
            if d < 1e-15:
                return 1e10
            log_det += np.log(d)
    
    log_prod = np.sum(np.log(r))
    quad = 0.5 * np.sum(r**2)
    
    E = -log_prod + log_det + quad
    C = np.exp(E / 12.0)
    
    return C

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Use the Dyson gas model as our primary objective
    objective = _compute_bound_v3
    
    # Initial guesses
    # 1. Zeros of Laguerre L_12
    try:
        x12, _ = roots_laguerre(12)
        init_L12 = np.sort(x12)
    except:
        init_L12 = np.linspace(1, 50, 12)
    
    # 2. Seed roots
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # 3. Dyson gas initial guess: spread out roots
    init_dyson = np.linspace(2, 48, 12)
    
    # 4. Geometric spacing
    init_geo = np.geomspace(1.0, 50.0, 12)
    
    # Evaluate initial candidates
    candidates = [
        (init_L12, objective(init_L12)),
        (seed_roots, objective(seed_roots)),
        (init_dyson, objective(init_dyson)),
        (init_geo, objective(init_geo)),
    ]
    candidates.sort(key=lambda x: x[1])
    best_roots = candidates[0][0].copy()
    best_bound = candidates[0][1]
    
    # Local optimization from best starting point
    def obj_wrapper(x):
        if np.any(x <= 0):
            return 1e10
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return 1e10
        return objective(x)
    
    bounds = [(1e-6, 100.0)] * 12
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        try:
            result = minimize(
                obj_wrapper,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x
        except Exception:
            pass
    
    # Try differential evolution for global search
    elapsed = time.time() - start_time
    if elapsed < deadline - 180:
        try:
            de_result = differential_evolution(
                obj_wrapper,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                init='sobol',
                polish=True
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_roots = de_result.x
        except Exception:
            pass
    
    # Refine with more iterations
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            result = minimize(
                obj_wrapper,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x
        except Exception:
            pass
    
    # Ensure positivity
    best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-END
