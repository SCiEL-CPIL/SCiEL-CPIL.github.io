"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre, roots_jacobi
from scipy.optimize import minimize, minimize_scalar
import time

def _compute_c4_bound_fast(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    Uses Gauss-Laguerre quadrature for numerical integration.
    Metric: int(P^2 * e^{-x}) / (int(P * e^{-x}))^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature (includes e^{-x} weight)
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2
    # Vectorized computation
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    # Clip to prevent overflow/underflow issues
    log_P = np.clip(log_P, -700, 700)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2) or np.isnan(int_P) or np.isnan(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _roots_from_gamma(gamma):
    """
    Generate 12 positive roots as nodes of generalized Laguerre quadrature
    for weight w(x) = e^{-x} * x^alpha * (1+x)^{-gamma} approximated via Jacobi.
    
    For gamma=0, this reduces to standard Laguerre roots.
    We use the fact that roots of L_n^{(alpha)}(x) correspond to alpha parameter.
    Here we map gamma to alpha to get a smooth family of root distributions.
    """
    # Map gamma to alpha: alpha = gamma - 1 (Laguerre L_n^{(alpha)})
    # For gamma in [0, 5], alpha in [-1, 4]
    # Use roots_jacobi with appropriate parameters to approximate generalized Laguerre
    # Actually, scipy has roots_laguerre only for standard. Let's use a different approach.
    # 
    # Better approach: Use the equilibrium condition from the Dyson gas model.
    # For weight e^{-x}(1+x)^{-gamma}, the equilibrium satisfies:
    # sum_j 1/(r_i - r_j) = 1/2 + gamma/(2(1+r_i))
    # (with factor of 2 for double roots, adjusted)
    #
    # For now, use a parametric family: r_k = A * (k+1)^p where p depends on gamma
    # This gives a smooth family from geometric to power-law spacing.
    
    k = np.arange(1, 13, dtype=np.float64)  # 1 to 12
    
    # For gamma=0: standard Laguerre-like (roughly linear spacing at large k)
    # For gamma>0: more spread out (heavier tail in weight means roots spread)
    # For gamma<0: more compact
    
    # Use power-law: r_k ~ k^{1+gamma/6} scaled appropriately
    # At gamma=0, r_k ~ k (linear), which approximates Laguerre asymptotics
    exponent = 1.0 + gamma / 6.0
    scale = 1.0 + 0.1 * gamma  # slight scaling
    
    roots = scale * k ** exponent
    roots = np.sort(roots)
    return roots

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def _objective_gamma(gamma):
    """Objective over single gamma parameter."""
    roots = _roots_from_gamma(gamma)
    if np.any(roots <= 0):
        return np.inf
    return _compute_c4_bound_fast(roots)

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Strategy:
    1. Coarse 1D search over gamma (generalized weight parameter)
    2. Full 12D refinement from best gamma-based starting point
    3. Diverse initializations including perturbed Laguerre roots
    4. High-precision polish
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Initial guess: Zeros of Laguerre L_12(x)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Coarse 1D search over gamma
    # gamma controls the tail decay of the effective weight
    # Optimal gamma likely in [-2, 5]
    gamma_result = minimize_scalar(
        _objective_gamma,
        bounds=(-3.0, 5.0),
        method='bounded',
        options={'xatol': 1e-6, 'maxiter': 100}
    )
    
    if gamma_result.success or gamma_result.fun < best_score:
        gamma_best = gamma_result.x
        roots_gamma = _roots_from_gamma(gamma_best)
        score_gamma = _compute_c4_bound_fast(roots_gamma)
        if score_gamma < best_score:
            best_score = score_gamma
            best_roots = roots_gamma.copy()
    
    # Phase 2: Generate diverse initial points for 12D optimization
    initial_points = [best_roots.copy()]
    
    # Perturbed Laguerre roots
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
        
    # Scaled Laguerre roots
    for scale in [0.7, 0.85, 1.1, 1.3, 1.5, 2.0]:
        scaled = x0 * scale
        initial_points.append(scaled.copy())
        
    # Gamma-based roots with perturbations
    if 'gamma_best' in locals():
        for scale in [0.5, 1.0, 1.5]:
            g_roots = _roots_from_gamma(gamma_best) * scale
            g_roots = np.maximum(g_roots, 1e-3)
            g_roots = np.sort(g_roots)
            initial_points.append(g_roots)
        
    # Geometric spacing variations
    for start in [0.5, 1.0, 2.0]:
        for ratio in [1.3, 1.5, 1.8, 2.0]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 200 and geom[0] > 0.1:
                initial_points.append(geom.copy())
                
    # Log-uniform spacing
    for log_start in [-2.0, -1.0, 0.0]:
        for log_end in [1.0, 2.0, 3.0, 4.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 200:
                    initial_points.append(roots)

    # Phase 3: Refine each initial point with L-BFGS-B
    bounds = [(1e-6, None)] * 12
    max_initials = 25  # Limit number of initial points to stay within time budget
    
    for i, x_init in enumerate(initial_points[:max_initials]):
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            candidate = result.x
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 15:
        try:
            result_polish = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_polish.x > 0) and not np.any(np.isnan(result_polish.x)):
                candidate = np.sort(result_polish.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Ensure the output is a list of 12 positive floats
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        result = [float(xi) for xi in x0]
        
    return result
# EVOLVE-BLOCK-END
