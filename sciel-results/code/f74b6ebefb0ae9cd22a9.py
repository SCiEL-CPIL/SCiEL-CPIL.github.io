"""8-node quadrature for luminosity distance using rational-function-based design."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _params_to_nodes_weights(params):
    raw_nodes = 1.0 / (1.0 + np.exp(-params[:8]))
    raw_nodes = np.sort(raw_nodes)
    for i in range(1, 8):
        if raw_nodes[i] <= raw_nodes[i-1]:
            raw_nodes[i] = raw_nodes[i-1] + 1e-10
    raw_nodes = np.clip(raw_nodes, 1e-10, 1.0 - 1e-10)
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    return raw_nodes, weights

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Test cosmologies
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (1.0, -2.0), (-2.0, 1.0),
        (0.0, 1.0), (-1.0, 1.0), (0.0, -1.0), (-0.5, -0.5),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    # Reference: 128-point Gauss-Legendre
    ref_x, ref_w = np.polynomial.legendre.leggauss(128)
    ref_nodes = 0.5 * (ref_x + 1.0)
    ref_weights = 0.5 * ref_w
    
    # Pre-compute true dL for all combinations
    true_dLs = np.zeros((len(test_cosmos), len(test_zs)))
    for i, (w0, wa) in enumerate(test_cosmos):
        for j, z in enumerate(test_zs):
            a_ref = 1.0 / (1.0 + z * ref_nodes)
            E_ref = _E_cpl(a_ref, w0, wa)
            ref_integral = np.sum(ref_weights / E_ref)
            true_dLs[i, j] = (1.0 + z) * z * ref_integral
    
    # Initial guess: Gauss-Legendre 8-point
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    params_init = np.zeros(16)
    params_init[:8] = np.log(nodes_gl / (1 - nodes_gl))
    params_init[8:16] = np.log(weights_gl)
    
    best_params = params_init.copy()
    
    def objective(params):
        if _time_left() < 5.0:
            return 1e10
        nodes, weights = _params_to_nodes_weights(params)
        
        max_err = 0.0
        for i, (w0, wa) in enumerate(test_cosmos):
            for j, z in enumerate(test_zs):
                a_8 = 1.0 / (1.0 + z * nodes)
                E_8 = _E_cpl(a_8, w0, wa)
                approx_integral = np.sum(weights / E_8)
                approx_dL = (1.0 + z) * z * approx_integral
                true_dL = true_dLs[i, j]
                if true_dL > 1e-10:
                    rel_err = abs(approx_dL - true_dL) / true_dL
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err
    
    best_score = objective(best_params)
    
    # Strategy: Use a multi-stage optimization
    # Stage 1: Coarse grid search over node patterns
    # Stage 2: Differential evolution
    # Stage 3: Local polish
    
    # Stage 1: Try different node distributions
    if _time_left() > 40.0:
        node_patterns = [
            # GL-like
            np.array([0.0506, 0.1512, 0.2715, 0.4083, 0.5917, 0.7285, 0.8488, 0.9494]),
            # More spread out
            np.array([0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
            # Concentrated near 0
            np.array([0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]),
            # Log-spaced
            np.logspace(np.log10(0.01), np.log10(0.99), 8),
            # sqrt-spaced
            np.linspace(0.01, 0.99, 8)**0.5,
            # 1/(1+x) transformed
            1.0 / (1.0 + np.linspace(0, 10, 8)),
        ]
        
        for pattern in node_patterns:
            pattern = np.clip(pattern, 1e-6, 1.0 - 1e-6)
            # Fit weights via least squares for each cosmo, then average
            all_weights = []
            for (w0, wa) in test_cosmos:
                for z in test_zs:
                    a_8 = 1.0 / (1.0 + z * pattern)
                    E_8 = _E_cpl(a_8, w0, wa)
                    target = true_dLs[np.where([c == (w0, wa) for c in test_cosmos])[0][0], 
                                      np.where(test_zs == z)[0][0]]
                    # We want: (1+z)*z*sum(w_i/E_i) = target
                    # sum(w_i/E_i) = target / ((1+z)*z)
                    rhs = target / ((1.0 + z) * z) if z > 0 else 0.0
                    A = 1.0 / E_8
                    # Solve A @ w = rhs, least squares
                    w_sol, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
                    w_sol = np.maximum(w_sol, 1e-10)
                    w_sol = w_sol / np.sum(w_sol)
                    all_weights.append(w_sol)
            
            # Average weights
            avg_weights = np.mean(all_weights, axis=0)
            avg_weights = avg_weights / np.sum(avg_weights)
            
            # Convert to params
            p = np.zeros(16)
            p[:8] = np.log(pattern / (1 - pattern))
            p[8:16] = np.log(avg_weights)
            
            score = objective(p)
            if score < best_score:
                best_score = score
                best_params = p
    
    # Stage 2: Differential Evolution
    if _time_left() > 15.0:
        bounds = [(-10, 10)] * 8 + [(-5, 5)] * 8
        try:
            res = differential_evolution(objective, bounds, maxiter=150, popsize=12,
                                         tol=1e-8, seed=42, polish=False,
                                         init='latinhypercube')
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Stage 3: Local polish
    if _time_left() > 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-15,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Stage 4: Fine-tune with Powell
    if _time_left() > 5.0:
        try:
            res = minimize(objective, best_params, method='Powell',
                          options={'maxiter': 500, 'xtol': 1e-14, 'ftol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    nodes, weights = _params_to_nodes_weights(best_params)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = weights / np.sum(weights)
    
    if not np.all(nodes < 1.0) or not np.all(nodes > 0.0):
        nodes, weights = nodes_gl, weights_gl
    
    return {"nodes": nodes, "weights": weights}
