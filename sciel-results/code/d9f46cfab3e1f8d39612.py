"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def _compute_c4_bound_fast(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    Uses Gauss-Laguerre quadrature for numerical integration.
    Metric: int(P^2 * w) / (int(P * w))^2 where P = prod (x - r_i)^2
    """
    roots = np.asarray(roots, dtype=float)
    # Use higher-order Gauss-Laguerre for better accuracy
    xq, wq = roots_laguerre(80)
    # P(x)^2 = prod (x - r_i)^4
    log_p_sq = np.zeros_like(xq)
    for r in roots:
        log_p_sq += 4.0 * np.log(np.abs(xq - r) + 1e-300)
    p_sq = np.exp(log_p_sq)
    weight = xq * np.exp(-xq)  # n=1
    int0 = np.sum(wq * p_sq * weight)
    int4 = np.sum(wq * p_sq * weight * xq ** 4)
    if int0 < 1e-30:
        return 1e30
    return int4 / int0

def _objective(roots):
    """Objective: C4 bound for given roots."""
    if np.any(roots <= 0) or np.any(roots > 1e6):
        return 1e30
    return _compute_c4_bound_fast(roots)

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Uses direct root optimization with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    best_roots = None
    best_score = 1e30
    
    # Initialize with various strategies
    initializations = []
    
    # Strategy 1: Uniformly spaced
    initializations.append(np.linspace(1.0, 24.0, 12))
    
    # Strategy 2: Log-spaced
    initializations.append(np.geomspace(1.0, 24.0, 12))
    
    # Strategy 3: Based on Laguerre polynomial zeros (scaled)
    lag_zeros, _ = roots_laguerre(12)
    initializations.append(lag_zeros + 1.0)
    
    # Strategy 4: Random positive roots
    np.random.seed(42)
    for _ in range(20):
        initializations.append(np.sort(np.random.uniform(0.5, 30.0, 12)))
    
    # Strategy 5: Clustered
    for center in [5.0, 10.0, 15.0]:
        for spread in [1.0, 3.0, 5.0]:
            initializations.append(np.sort(center + spread * np.random.randn(12)))
    
    # Evaluate all initializations and keep best few for refinement
    candidates = []
    for init_roots in initializations:
        if time.time() > deadline - 30:
            break
        try:
            score = _objective(init_roots)
            if score < best_score:
                best_score = score
                best_roots = init_roots.copy()
            candidates.append((score, init_roots.copy()))
        except Exception:
            pass
    
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates with local optimization
    for i, (score, roots) in enumerate(candidates[:10]):
        if time.time() > deadline - 30:
            break
        if score >= best_score:
            continue
        
        try:
            res = minimize(
                _objective,
                roots,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_roots = res.x.copy()
        except Exception:
            pass
    
    # Additional random restarts with refinement
    np.random.seed(123)
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        init_roots = np.sort(np.random.uniform(0.5, 30.0, 12))
        try:
            score = _objective(init_roots)
            if score < best_score:
                best_score = score
                best_roots = init_roots.copy()
            
            # Quick local refinement
            res = minimize(
                _objective,
                init_roots,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_roots = res.x.copy()
        except Exception:
            pass
    
    if best_roots is None:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
    
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        result = [4.0 * (i + 1) for i in range(12)]
    return result
# EVOLVE-BLOCK-END
