"""Suite task: universal 8-node positive quadrature for luminosity distance.

Strategy: Moment matching against a rich family of CPL cosmologies.
Instead of a general-purpose global optimizer like Differential Evolution which
can be slow and noisy, we use a structured approach:
1. Define a dense grid of CPL models (w0, wa).
2. Compute reference integrals using high-precision adaptive quadrature.
3. Use a robust local optimization (L-BFGS-B with multi-start) to minimize the
   worst-case relative error. The parameterization uses logarithmic gaps for nodes
   and log-weights to ensure positivity and strict ordering.
4. The key insight from inspirations is that moment matching (fitting the quadrature
   to the specific shape of the integrands) outperforms polynomial-optimal rules
   like Gauss-Legendre for these non-polynomial cosmological integrands.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 85.0  # Safety margin before 90s kill

def _time_ok():
    return time.time() < _DEADLINE

def _E(a, w0, wa):
    """
    E(a) = H(a)/H0 for spatially flat CPL model.
    The seed program uses:
    E(a) = a^-3 + a^(-3(1+w0+wa)) / (1+w0+wa)
    
    This corresponds to a specific parametrization where the dark energy term
    behaves like a power law with effective w = w0 + wa.
    We MUST match this exactly to align with the evaluator's definition implied by the seed.
    """
    w_eff = w0 + wa
    denom = 1.0 + w_eff
    # Handle singularity if denom is close to 0
    if abs(denom) < 1e-12:
        # Limit as w_eff -> -1 is log(a) * a^-3 ? No, integral of a^-3(1+w) is a^-3w / (-3w).
        # If w_eff = -1, term is a^0 / 0 -> infinity? 
        # Actually, the term is a^(-3(1+w_eff)) / (1+w_eff).
        # If w_eff = -1, it's a^0 / 0. This is a singular case.
        # The seed skips models where |w0+1+wa| < 1e-4.
        # So we assume denom is safe in the models we evaluate.
        return np.inf
    return a ** (-3.0) + (a ** (-3.0 * w_eff)) / denom

def _integrand(x, w0, wa):
    """Integrand over x in [0,1]: 1 / E(a(x)) with a(x) = 1/(1+x)."""
    a = 1.0 / (1.0 + x)
    return 1.0 / _E(a, w0, wa)

def _true_integral(w0, wa):
    """Reference integral via high-order adaptive Gauss-Kronrod."""
    val, _ = quad(_integrand, 0.0, 1.0, args=(w0, wa), limit=200,
                  epsabs=1e-13, epsrel=1e-13)
    return val

def _build_models():
    """
    Build a diverse set of CPL models.
    We use a grid that covers the physically relevant range.
    """
    w0_list = np.linspace(-1.5, 0.5, 11)
    wa_list = np.linspace(-0.5, 0.5, 11)
    models = []
    for w0 in w0_list:
        for wa in wa_list:
            if abs(w0 + 1.0 + wa) < 1e-4:
                continue
            try:
                I = _true_integral(w0, wa)
            except Exception:
                continue
            if not np.isfinite(I) or I <= 0.0:
                continue
            models.append((w0, wa, I))
    return models

# Pre-compute models and integrals
_MODELS = _build_models()
if not _MODELS:
    _MODELS = [(-0.5, 0.0, _true_integral(-0.5, 0.0))]

def _nodes_from_params(p):
    """
    Map parameters to strictly increasing nodes in (0,1).
    p[0] = logit(node_0)
    p[i] = logit( (node_i - node_{i-1}) / (1 - node_{i-1}) ) for i > 0
    This ensures 0 < n_0 < n_1 < ... < n_7 < 1.
    """
    nodes = np.zeros(8)
    nodes[0] = 1.0 / (1.0 + np.exp(-p[0]))
    for i in range(1, 8):
        gap = 1.0 / (1.0 + np.exp(-p[i]))
        # gap is in (0,1), representing fraction of remaining space
        nodes[i] = nodes[i-1] + gap * (1.0 - nodes[i-1])
    return nodes

def _weights_from_params(p):
    """
    Map parameters to positive weights summing to 1.
    p[8:16] are log(weights).
    """
    w_raw = np.exp(p[8:16])
    return w_raw / w_raw.sum()

def _quadrature(nodes, weights, w0, wa):
    a = 1.0 / (1.0 + nodes)
    w_eff = w0 + wa
    denom = 1.0 + w_eff
    E = a ** (-3.0) + (a ** (-3.0 * w_eff)) / denom
    return float(np.dot(weights, 1.0 / E))

def _objective(p):
    """Minimize the worst-case relative error across all models."""
    if not _time_ok():
        return 1.0
    
    nodes = _nodes_from_params(p)
    weights = _weights_from_params(p)
    
    # Check validity
    if not np.all(np.diff(nodes) > 1e-12):
        return 1.0
    if not np.all(weights > 0):
        return 1.0
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
        return 1.0
        
    max_err = 0.0
    for w0, wa, I in _MODELS:
        q = _quadrature(nodes, weights, w0, wa)
        if not np.isfinite(q):
            return 1.0
        err = abs(q - I) / I
        if err > max_err:
            max_err = err
            
    return max_err

def _generate_initial_guesses(num_starts=10):
    """
    Generate diverse initial guesses for multi-start optimization.
    1. Gauss-Legendre
    2. Gauss-Lobatto (if available or approximated)
    3. Uniform nodes
    4. Random smooth perturbations of GL
    """
    guesses = []
    
    # 1. Gauss-Legendre
    x, w = np.polynomial.legendre.leggauss(8)
    nodes_gl = (x + 1.0) / 2.0
    weights_gl = w / 2.0
    
    # Convert GL nodes to param space
    t = np.zeros(8)
    t[0] = np.log(nodes_gl[0] / (1 - nodes_gl[0]))
    for i in range(1, 8):
        gap = (nodes_gl[i] - nodes_gl[i-1]) / (1 - nodes_gl[i-1])
        t[i] = np.log(gap / (1 - gap))
    w_log = np.log(weights_gl)
    p_gl = np.concatenate([t, w_log])
    guesses.append(p_gl)
    
    # 2. Uniform nodes
    nodes_unif = np.linspace(1.0/16.0, 15.0/16.0, 8)
    weights_unif = np.ones(8) / 8.0
    
    t_unif = np.zeros(8)
    t_unif[0] = np.log(nodes_unif[0] / (1 - nodes_unif[0]))
    for i in range(1, 8):
        gap = (nodes_unif[i] - nodes_unif[i-1]) / (1 - nodes_unif[i-1])
        t_unif[i] = np.log(gap / (1 - gap))
    w_log_unif = np.log(weights_unif)
    p_unif = np.concatenate([t_unif, w_log_unif])
    guesses.append(p_unif)
    
    # 3-10. Perturbed GL
    rng = np.random.RandomState(42)
    for i in range(num_starts - 2):
        p_pert = p_gl.copy()
        # Small random perturbations in param space
        p_pert += rng.normal(0, 0.1, size=16)
        guesses.append(p_pert)
        
    return guesses

def construct():
    """
    Fit a universal 8-node positive quadrature rule using multi-start L-BFGS-B.
    """
    best_p = None
    best_err = float('inf')
    
    guesses = _generate_initial_guesses(num_starts=10)
    
    for p0 in guesses:
        if not _time_ok():
            break
            
        try:
            res = minimize(
                _objective, 
                p0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_err:
                best_err = res.fun
                best_p = res.x
                
        except Exception:
            continue
            
    if best_p is None:
        # Fallback to Gauss-Legendre
        x, w = np.polynomial.legendre.leggauss(8)
        nodes_gl = (x + 1.0) / 2.0
        weights_gl = w / 2.0
        return {"nodes": nodes_gl, "weights": weights_gl}
    
    nodes = _nodes_from_params(best_p)
    weights = _weights_from_params(best_p)
    
    # Final polish: try a few more iterations from the best point
    if _time_ok():
        try:
            res2 = minimize(
                _objective, 
                best_p, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.fun < best_err:
                best_p = res2.x
                nodes = _nodes_from_params(best_p)
                weights = _weights_from_params(best_p)
        except Exception:
            pass

    # Ensure weights sum to 1 exactly
    weights = weights / weights.sum()
    
    # Final validity check
    if not (np.all(np.diff(nodes) > 1e-12) and np.all(nodes > 0) and np.all(nodes < 1) and np.all(weights > 0)):
         # Fallback
         x, w = np.polynomial.legendre.leggauss(8)
         nodes_gl = (x + 1.0) / 2.0
         weights_gl = w / 2.0
         return {"nodes": nodes_gl, "weights": weights_gl}
         
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
