"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_norm_log_and_grad(roots, n_quad=500):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)^2
    (double roots) with weight e^{-x}, and its gradient w.r.t. roots.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound.
    """
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log P(x)^2 = 2 * sum log|x - r_i| for double roots
    # P(x) = prod (x - r_i)^2, so log|P(x)| = 2 * sum log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 2.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
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
    
    # Gradient: d/d(r_j) log_norm_sq
    # d log_norm_sq / d r_j = 2 * sum_i w_i P(x_i)^2 e^{-x_i} * (1/(x_i - r_j)) / norm_sq
    # = 2 * sum_i weighted_i * (1/(x_i - r_j))
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        diff[np.abs(diff) < 1e-300] = 1e-300
        grad[j] = 2.0 * np.sum(weighted / diff)
    
    return log_norm_sq, grad

def _dyson_gas_init_double(m=12):
    """
    Initialize root positions using Dyson gas equilibrium with repulsion exponent 4
    (double roots -> factor 2 in log derivative, so repulsion is 2*2=4 per pair)
    and linear confining potential.
    
    For double roots, the equilibrium condition is:
    sum_{j != i} 4/(x_i - x_j) = x_i
    """
    roots = np.linspace(2.0, 45.0, m)
    
    for iteration in range(1000):
        forces = np.zeros(m)
        for i in range(m):
            forces[i] = roots[i]
            for j in range(m):
                if i != j:
                    forces[i] -= 4.0 / (roots[i] - roots[j])
        
        step_size = 0.005
        for i in range(m):
            roots[i] += step_size * forces[i]
            roots[i] = max(roots[i], 0.3)
        
        if np.max(np.abs(forces)) < 1e-12:
            break
    
    return np.sort(roots)

def _laguerre_roots_init(m=12):
    """
    Use the first m zeros of the Laguerre polynomial L_{2m}(x) as initial guess.
    These are natural for Laguerre-weighted problems.
    """
    from scipy.special import roots_laguerre
    x, _ = roots_laguerre(2 * m)
    # Take the first m positive roots (they're all positive)
    return np.sort(x[:m])

def _geometric_init(m=12):
    """Geometric spacing initialization."""
    r0 = 3.0
    q = 1.25
    roots = r0 * (q ** np.arange(m))
    return np.sort(roots)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 595.0
    
    def time_left():
        return deadline - time.time()
    
    # Generate multiple initializations
    inits = []
    
    try:
        inits.append(_dyson_gas_init_double(12))
    except Exception:
        pass
    
    try:
        inits.append(_laguerre_roots_init(12))
    except Exception:
        pass
    
    try:
        inits.append(_geometric_init(12))
    except Exception:
        pass
    
    # Seed roots as fallback
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    inits.append(seed_roots)
    
    # Evaluate all initializations
    best_roots = None
    best_log_norm = -np.inf
    
    for roots in inits:
        if time_left() < 30:
            break
        try:
            log_norm, _ = _compute_norm_log_and_grad(roots, n_quad=500)
            if np.isfinite(log_norm) and log_norm > best_log_norm:
                best_log_norm = log_norm
                best_roots = roots.copy()
        except Exception:
            pass
    
    if best_roots is None:
        best_roots = seed_roots.copy()
        best_log_norm, _ = _compute_norm_log_and_grad(best_roots, n_quad=500)
    
    def objective(params, n_quad=500):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            return 1e10
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10
        log_norm, _ = _compute_norm_log_and_grad(roots, n_quad=n_quad)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    def objective_grad(params, n_quad=500):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            return 1e10, np.zeros(len(params))
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10, np.zeros(len(params))
        log_norm, grad = _compute_norm_log_and_grad(roots, n_quad=n_quad)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(len(params))
        return -log_norm, -grad
    
    # Optimization from best initialization
    try:
        result = minimize(
            lambda p: objective(p, 500),
            best_roots,
            jac=lambda p: objective_grad(p, 500),
            method='L-BFGS-B',
            bounds=[(0.3, None)] * 12,
            options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if np.isfinite(result.fun) and result.fun < -best_log_norm:
            best_roots = np.sort(result.x)
            best_log_norm = -result.fun
    except Exception:
        pass
    
    # Try other initializations if time permits
    for roots in inits:
        if time_left() < 60:
            break
        if np.allclose(roots, best_roots, atol=1e-6):
            continue
        try:
            result = minimize(
                lambda p: objective(p, 500),
                roots,
                jac=lambda p: objective_grad(p, 500),
                method='L-BFGS-B',
                bounds=[(0.3, None)] * 12,
                options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if np.isfinite(result.fun) and result.fun < -best_log_norm:
                best_roots = np.sort(result.x)
                best_log_norm = -result.fun
        except Exception:
            pass
    
    # Final high-precision polish with more quadrature points
    if time_left() > 40:
        try:
            result = minimize(
                lambda p: objective(p, 800),
                best_roots,
                jac=lambda p: objective_grad(p, 800),
                method='L-BFGS-B',
                bounds=[(0.3, None)] * 12,
                options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.isfinite(result.fun) and result.fun < -best_log_norm:
                best_roots = np.sort(result.x)
                best_log_norm = -result.fun
        except Exception:
            pass
    
    # Another polish with even more quadrature if time permits
    if time_left() > 25:
        try:
            result = minimize(
                lambda p: objective(p, 1200),
                best_roots,
                jac=lambda p: objective_grad(p, 1200),
                method='L-BFGS-B',
                bounds=[(0.3, None)] * 12,
                options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if np.isfinite(result.fun) and result.fun < -best_log_norm:
                best_roots = np.sort(result.x)
                best_log_norm = -result.fun
        except Exception:
            pass
    
    # Final verification
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = seed_roots.copy()
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
