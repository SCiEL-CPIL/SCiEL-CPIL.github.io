"""Optimized 8-node positive quadrature for luminosity distance."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _E_cpl(a, w0, wa, om_m=0.3, om_r=5e-5):
    """E(a) = H(a)/H0 for flat CPL model."""
    om_de = 1.0 - om_m - om_r
    # E^2 = om_m * a^-3 + om_r * a^-4 + om_de * a^(-3*(1+w0+wa)) * (1 + wa*(1-a))
    e2 = om_m * a**(-3) + om_r * a**(-4) + om_de * a**(-3.0*(1.0 + w0 + wa)) * (1.0 + wa * (1.0 - a))
    return np.sqrt(np.maximum(e2, 1e-30))

def _integrand_z(z, w0, wa, om_m=0.3, om_r=5e-5):
    """Integrand in z-space: z/(1+z) * 1/E(a) where a=1/(1+z)."""
    a = 1.0 / (1.0 + z)
    E = _E_cpl(a, w0, wa, om_m, om_r)
    return (z / (1.0 + z)) / E

def _true_integral(w0, wa, om_m=0.3, om_r=5e-5):
    """High-precision adaptive integral of the z-space integrand."""
    val, _ = quad(_integrand_z, 0.0, 1.0, args=(w0, wa, om_m, om_r), limit=200, epsabs=1e-14, epsrel=1e-14)
    return val

def _quad_value(nodes, weights, w0, wa, om_m=0.3, om_r=5e-5):
    """Quadrature approximation using given nodes and weights."""
    vals = np.array([_integrand_z(z, w0, wa, om_m, om_r) for z in nodes])
    return np.dot(weights, vals)

def _objective(params, models):
    """Worst-case relative error across models."""
    n = 8
    log_w = params[:n]
    raw_nodes = params[n:]
    
    weights = np.exp(log_w)
    weights = weights / weights.sum()
    
    # Map raw_nodes to strictly increasing in (0,1) using sigmoid-like transform
    # Use cumulative sum of exponentials to ensure strict increase
    diffs = np.exp(raw_nodes)
    total = diffs.sum()
    nodes = np.cumsum(diffs) / total
    # Ensure strictly in (0,1)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strict increase
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes[-1] = min(nodes[-1], 1.0 - 1e-10)
    
    worst_err = 0.0
    for (w0, wa, om_m, om_r) in models:
        true_val = _true_integral(w0, wa, om_m, om_r)
        if abs(true_val) < 1e-15:
            continue
        quad_val = _quad_value(nodes, weights, w0, wa, om_m, om_r)
        rel_err = abs(quad_val - true_val) / abs(true_val)
        worst_err = max(worst_err, rel_err)
    return worst_err

def construct():
    if _remaining() < 2.0:
        # Fallback: midpoint rule
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    # Generate diverse training cosmologies
    models = []
    # Standard LambdaCDM
    models.append((0.0, 0.0, 0.3, 5e-5))
    # Varying w0
    for w0 in [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]:
        models.append((w0, 0.0, 0.3, 5e-5))
    # Varying wa
    for wa in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        models.append((0.0, wa, 0.3, 5e-5))
    # Combined
    for w0, wa in [(-1.5, 1.0), (-1.0, 0.5), (-0.5, -0.5), (0.0, 1.0), (0.5, -1.0)]:
        models.append((w0, wa, 0.3, 5e-5))
    # Varying matter fraction
    for om_m in [0.2, 0.25, 0.35, 0.4]:
        models.append((0.0, 0.0, om_m, 5e-5))
    
    # Remove duplicates
    seen = set()
    unique_models = []
    for m in models:
        key = (round(m[0],6), round(m[1],6), round(m[2],6), round(m[3],6))
        if key not in seen:
            seen.add(key)
            unique_models.append(m)
    models = unique_models
    
    n = 8
    # Initial guess: midpoint nodes, equal weights
    init_nodes = (np.arange(n) + 0.5) / n
    init_weights = np.full(n, 1.0 / n)
    
    # Parameterize: log(weights) for positivity, raw diffs for nodes
    log_w0 = np.log(init_weights)
    # For nodes, use log-differences to ensure strict increase
    diffs = np.diff(np.concatenate([[0], init_nodes, [1]]))
    raw_nodes0 = np.log(diffs[1:-1])  # log of internal diffs
    
    x0 = np.concatenate([log_w0, raw_nodes0])
    
    best_params = x0.copy()
    best_obj = _objective(x0, models)
    
    # Try Nelder-Mead from multiple starting points
    rng = np.random.default_rng(42)
    
    n_starts = 5
    for i in range(n_starts):
        if _remaining() < 3.0:
            break
        
        if i == 0:
            x_start = x0.copy()
        else:
            # Perturb initial guess
            x_start = x0 + rng.normal(0, 0.1, size=x0.shape)
        
        try:
            res = minimize(_objective, x_start, args=(models,), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if _remaining() < 5.0:
            break
    
    # Extract final nodes and weights
    log_w = best_params[:n]
    raw_nodes = best_params[n:]
    
    weights = np.exp(log_w)
    weights = weights / weights.sum()
    
    diffs = np.exp(raw_nodes)
    total = diffs.sum()
    nodes = np.cumsum(diffs) / total
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes[-1] = min(nodes[-1], 1.0 - 1e-10)
    
    # Final validation: ensure weights sum to 1 within 1e-12
    weights = weights / weights.sum()
    
    # Check if we improved over midpoint
    mid_nodes = (np.arange(n) + 0.5) / n
    mid_weights = np.full(n, 1.0 / n)
    
    worst_opt = 0.0
    worst_mid = 0.0
    for (w0, wa, om_m, om_r) in models:
        true_val = _true_integral(w0, wa, om_m, om_r)
        if abs(true_val) < 1e-15:
            continue
        q_opt = _quad_value(nodes, weights, w0, wa, om_m, om_r)
        q_mid = _quad_value(mid_nodes, mid_weights, w0, wa, om_m, om_r)
        worst_opt = max(worst_opt, abs(q_opt - true_val) / abs(true_val))
        worst_mid = max(worst_mid, abs(q_mid - true_val) / abs(true_val))
    
    if worst_opt <= worst_mid:
        return {"nodes": nodes, "weights": weights}
    else:
        return {"nodes": mid_nodes, "weights": mid_weights}
# EVOLVE-BLOCK-END
