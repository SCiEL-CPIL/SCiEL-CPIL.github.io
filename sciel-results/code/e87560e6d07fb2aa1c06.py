"""Refined uncertainty inequality: Gauss-Laguerre node parameterization for C4 upper bound."""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, roots_genlaguerre

_START = time.time()
_DEADLINE = 630.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - _MARGIN - time.time()

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 positive root locations.
    The bound is 1 / sqrt(||P||^2), where P(x) = prod (x - r_i)^2.
    ||P||^2 is the integral of P(x)^2 * x^2 * exp(-x) dx.
    """
    if not np.all(np.isfinite(roots)) or np.any(roots <= 0):
        return 1e100
    
    # Build Q(x) = prod (x - r_i)
    q_coeff = np.array([1.0])
    for r in roots:
        q_coeff = np.convolve(q_coeff, np.array([1.0, -r]))
        
    # P(x) = Q(x)^2
    p_coeff = np.convolve(q_coeff, q_coeff)
    
    # Compute norm squared: sum_{i,j} p_i p_j Gamma(i+j+3)
    # alpha = 2, so weight is x^2 exp(-x).
    # Integral x^{i+j+2} exp(-x) = Gamma(i+j+3)
    n = len(p_coeff)
    indices = np.arange(n)
    gamma_args = indices[:, None] + indices[None, :] + 3.0
    
    # Use log-gamma for stability
    log_gamma_vals = gammaln(gamma_args)
    log_gamma_vals = np.clip(log_gamma_vals, -700, 700)
    gamma_vals = np.exp(log_gamma_vals)
    
    # norm_sq = p_coeff^T G p_coeff
    norm_sq = p_coeff @ (gamma_vals @ p_coeff)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e100
        
    return 1.0 / np.sqrt(norm_sq)

def _gauss_laguerre_roots(alpha):
    """
    Get 12 roots of the generalized Laguerre polynomial L_12^alpha(x).
    These are the nodes of the Gauss-Laguerre quadrature for weight x^alpha exp(-x).
    """
    try:
        nodes, _ = roots_genlaguerre(12, alpha)
        # Ensure all nodes are positive (they should be for alpha > -1)
        if np.any(nodes <= 0) or not np.all(np.isfinite(nodes)):
            return None
        return nodes
    except Exception:
        return None

def _objective_alpha(alpha_val):
    """
    Objective: minimize C4 bound by varying the Laguerre exponent alpha.
    """
    if alpha_val <= -0.5:
        return 1e100
    roots = _gauss_laguerre_roots(alpha_val)
    if roots is None:
        return 1e100
    return _compute_c4_bound(roots)

def _objective_log_roots(log_roots):
    """
    Fine-tuning objective on log-space roots.
    """
    try:
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(roots > 1e6) or np.any(roots < 1e-6):
            return 1e100
        return _compute_c4_bound(roots)
    except Exception:
        return 1e100

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the upper bound on C4.
    Returns a list of 12 positive root locations.
    """
    deadline = _START + _DEADLINE - _MARGIN
    
    best_roots = None
    best_score = 1e100
    
    # Phase 1: 1D search over alpha (the Laguerre exponent)
    # The roots of L_12^alpha(x) give a natural 1-parameter family of 12-node configurations.
    # Scan a range of alpha values to find a good starting point.
    
    # Coarse scan over alpha in [0, 20]
    alpha_scan = np.linspace(0.0, 20.0, 100)
    for a in alpha_scan:
        if time.time() > deadline:
            break
        score = _objective_alpha(a)
        if score < best_score:
            best_score = score
            best_roots = _gauss_laguerre_roots(a)
    
    # Fine 1D optimization around the best alpha found
    if best_roots is not None:
        # Find the best alpha from the scan
        best_alpha = None
        for a in alpha_scan:
            if time.time() > deadline:
                break
            score = _objective_alpha(a)
            if score < best_score:
                best_score = score
                best_alpha = a
                best_roots = _gauss_laguerre_roots(a)
        
        if best_alpha is not None:
            # Refine with minimize_scalar
            try:
                res = minimize_scalar(
                    _objective_alpha,
                    bounds=(max(0.0, best_alpha - 2.0), best_alpha + 2.0),
                    method='bounded',
                    options={'xatol': 1e-10, 'maxiter': 200}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_roots = _gauss_laguerre_roots(res.x)
            except Exception:
                pass
    
    # Phase 2: Fine-tune the 12 roots in log-space starting from the best Gauss-Laguerre nodes
    if best_roots is not None and time.time() < deadline:
        log_best = np.log(best_roots)
        bounds = [(np.log(0.01), np.log(1000.0))] * 12
        
        # Multiple restarts with small perturbations
        np.random.seed(42)
        num_fine_starts = 10
        fine_starts = [log_best]
        for _ in range(num_fine_starts):
            perturbation = np.random.normal(0, 0.1, size=12)
            fine_starts.append(log_best + perturbation)
        
        for x0 in fine_starts:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    _objective_log_roots,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_roots = np.exp(result.x)
            except Exception:
                continue
    
    # Phase 3: Final refinement if time permits
    if best_roots is not None and time.time() < deadline - 60:
        try:
            log_best = np.log(best_roots)
            bounds = [(np.log(0.01), np.log(1000.0))] * 12
            result = minimize(
                _objective_log_roots,
                log_best,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_roots = np.exp(result.x)
        except Exception:
            pass
    
    if best_roots is None:
        # Fallback: use standard Laguerre nodes (alpha=0)
        best_roots = _gauss_laguerre_roots(0.0)
        if best_roots is None:
            best_roots = np.linspace(1.0, 12.0, 12)
    
    return [float(r) for r in best_roots]
