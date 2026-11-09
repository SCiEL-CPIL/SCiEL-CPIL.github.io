"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, genlaguerre
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def _compute_bound(roots):
    """Compute the upper bound on C4 given 12 double-root locations."""
    roots = np.asarray(roots, dtype=np.float64)
    m = 12
    n = 1
    
    if np.any(roots <= 0) or len(roots) != 12:
        return np.inf
    
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-12):
        return np.inf
    
    # The refined Cohn-Goncalves bound: for the Laguerre construction with
    # double roots at r_1,...,r_m, the upper bound on C4 is:
    # C4 <= ||L_m^{(0)}||_2^2 / prod_{i=1}^m |L_m^{(0)'}(r_i)| * correction
    # 
    # A more standard form for sign uncertainty: the bound is related to
    # the maximum of |f| where f is constructed from the Laguerre polynomial
    # divided by the product of (x - r_i)^2.
    #
    # For the specific Cohn-Goncalves construction, the bound on the
    # Fourier sign uncertainty constant C4 with m sign changes is:
    # C4 <= sqrt( (L_m(0))^2 * m! / (prod_{i=1}^m r_i^2 * prod_{i=1}^m |L_m'(r_i)|^2) )
    #
    # Let me use a cleaner formulation. The key quantity is:
    # bound = sqrt( L_m(0)^2 / (prod_{i=1}^m (r_i * L_m'(r_i)^2)) )
    #
    # Actually, for the standard result, the bound is:
    # C4 <= 1 / sqrt( prod_{i=1}^m |L_m'(r_i)|^2 / r_i )
    #
    # Let me compute this properly.
    
    alpha = n - 1  # = 0
    
    # L_m^{(alpha)}(0) = binom(m+alpha, m) = 1 for alpha=0
    Lm_0 = 1.0
    
    # Compute the derivative of L_m^{(0)} at each root
    # L_m'(x) = -L_{m-1}^{(1)}(x) for standard Laguerre
    # More generally: d/dx L_m^{(alpha)}(x) = -L_{m-1}^{(alpha+1)}(x)
    
    log_bound = 0.0
    for r in roots:
        # L_m'(r) = -L_{m-1}^{(1)}(r)
        deriv = -eval_genlaguerre(m - 1, alpha + 1, r)
        if abs(deriv) < 1e-30:
            return np.inf
        # Each factor contributes: log(r * deriv^2) in denominator
        log_bound += 0.5 * (np.log(r) + 2.0 * np.log(abs(deriv)))
    
    bound = np.exp(-log_bound)
    
    return bound

def construct():
    """Construct 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill
    
    m = 12
    alpha = 0  # n-1 where n=1
    
    # The natural choice: roots are the zeros of L_m^{(alpha)}(x)
    # For alpha=0, these are the standard Laguerre zeros
    try:
        from numpy.polynomial.laguerre import laggauss
        nodes, weights = laggauss(m)
        x0 = nodes.copy()
    except:
        # Fallback: compute zeros via companion matrix
        coeffs = genlaguerre(m, alpha)
        x0 = np.roots(coeffs)
        x0 = x0[np.isreal(x0)].real
        x0 = np.sort(x0)
    
    best_x = x0.copy()
    best_bound = _compute_bound(x0)
    
    # Optimize: minimize the bound by adjusting root positions
    # Use log-space parameterization to ensure positivity
    log_roots = np.log(x0)
    
    def objective(log_r):
        if time.time() > deadline:
            return np.inf
        roots = np.exp(log_r)
        return _compute_bound(roots)
    
    # Local optimization from the Laguerre zeros
    result = minimize(
        objective,
        log_roots,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    if result.success and result.fun < best_bound:
        best_x = np.exp(result.x)
        best_bound = result.fun
    
    # Try multiple random restarts in log-space
    rng = np.random.RandomState(42)
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Perturb the log-roots
        scale = 0.5 * (1 - i / n_restarts)
        perturbation = rng.randn(m) * scale
        log_r_init = log_roots + perturbation
        
        res = minimize(
            objective,
            log_r_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if res.success and res.fun < best_bound:
            best_x = np.exp(res.x)
            best_bound = res.fun
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        log_best = np.log(best_x)
        res = minimize(
            objective,
            log_best,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if res.success and res.fun < best_bound:
            best_x = np.exp(res.x)
            best_bound = res.fun
    
    # Ensure positivity and distinctness
    best_x = np.sort(best_x)
    if np.any(best_x <= 0):
        best_x = x0.copy()
    
    return list(best_x)
# EVOLVE-BLOCK-END
