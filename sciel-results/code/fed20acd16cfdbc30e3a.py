import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    def E(a, Om, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        if abs(w0 + 1.0) < 1e-8 and abs(wa) < 1e-8:
            return np.sqrt(Om * a**(-3.0) + (1.0 - Om))
        if abs(w0 + 1.0) < 1e-8:
            eps = 1e-8
            w0_eff = -1.0 + eps
            return np.sqrt(Om * a**(-3.0) + (1.0 - Om) * a**(-3.0 * (1.0 + w0_eff + wa)) * np.exp(3.0 * wa * (1.0 - a) / (1.0 + w0_eff)))
        
        exponent = -3.0 * (1.0 + w0 + wa)
        exp_term = 3.0 * wa * (1.0 - a) / (1.0 + w0)
        exp_term = np.clip(exp_term, -500.0, 500.0)
        term2 = (1.0 - Om) * a**exponent * np.exp(exp_term)
        return np.sqrt(Om * a**(-3.0) + term2)
    
    # Diverse test cosmologies covering parameter space
    models = [
        {'Om': 0.3, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.3, 'w0': -0.9, 'wa': 0.1},
        {'Om': 0.2, 'w0': -1.1, 'wa': 0.2},
        {'Om': 0.4, 'w0': -0.8, 'wa': -0.2},
        {'Om': 0.25, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.35, 'w0': -1.2, 'wa': 0.3},
        {'Om': 0.15, 'w0': -0.7, 'wa': -0.3},
        {'Om': 0.3, 'w0': -1.05, 'wa': 0.15},
        {'Om': 0.28, 'w0': -0.95, 'wa': 0.05},
        {'Om': 0.32, 'w0': -1.02, 'wa': -0.08},
    ]
    
    redshifts = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    
    # Pre-compute true integrals using quad
    true_integrals = np.zeros((len(models), len(redshifts)))
    for i, m in enumerate(models):
        Om, w0, wa = m['Om'], m['w0'], m['wa']
        for j, z in enumerate(redshifts):
            def integrand(xi, Om=Om, w0=w0, wa=wa, z=z):
                a = 1.0 / (1.0 + z * xi)
                return 1.0 / E(a, Om, w0, wa)
            val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_integrals[i, j] = val
    
    n_models = len(models)
    n_z = len(redshifts)
    
    # Vectorized E computation for all models at given nodes
    def E_all(a_vals, Om, w0, wa):
        """Vectorized E computation."""
        if abs(w0 + 1.0) < 1e-8 and abs(wa) < 1e-8:
            return np.sqrt(Om * a_vals**(-3.0) + (1.0 - Om))
        if abs(w0 + 1.0) < 1e-8:
            eps = 1e-8
            w0_eff = -1.0 + eps
            return np.sqrt(Om * a_vals**(-3.0) + (1.0 - Om) * a_vals**(-3.0 * (1.0 + w0_eff + wa)) * np.exp(3.0 * wa * (1.0 - a_vals) / (1.0 + w0_eff)))
        
        exponent = -3.0 * (1.0 + w0 + wa)
        exp_term = 3.0 * wa * (1.0 - a_vals) / (1.0 + w0)
        exp_term = np.clip(exp_term, -500.0, 500.0)
        term2 = (1.0 - Om) * a_vals**exponent * np.exp(exp_term)
        return np.sqrt(Om * a_vals**(-3.0) + term2)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Check validity
        if not np.all(nodes > 0) or not np.all(nodes < 1):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-10):
            return 1e10
        if not np.all(weights > 1e-10):
            return 1e10
        
        w_sum = np.sum(weights)
        if abs(w_sum - 1.0) > 1e-6:
            return 1e10
        
        # Compute relative errors for all model-redshift pairs
        errors = np.zeros((n_models, n_z))
        for i in range(n_models):
            Om, w0, wa = models[i]['Om'], models[i]['w0'], models[i]['wa']
            for j in range(n_z):
                z = redshifts[j]
                a_vals = 1.0 / (1.0 + z * nodes)
                e_vals = E_all(a_vals, Om, w0, wa)
                q_val = np.dot(weights, 1.0 / e_vals)
                true_val = true_integrals[i, j]
                errors[i, j] = abs(q_val - true_val) / true_val
        
        # Soft-max objective (smooth approximation to max)
        beta = 50.0
        max_err = np.log(np.sum(np.exp(beta * errors)) / (n_models * n_z)) / beta
        return max_err
    
    # Initial guess: Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    best_x = np.concatenate([nodes_gl, weights_gl])
    best_val = objective(best_x)
    
    bounds = [(1e-6, 1.0 - 1e-6)] * 8 + [(1e-8, 1.0)] * 8
    
    def callback(xk):
        nonlocal best_x, best_val
        val = objective(xk)
        if val < best_val:
            best_x = xk.copy()
            best_val = val
        return time.time() > deadline
    
    # Multiple restarts with different initial conditions
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if r == 0:
            x0 = best_x.copy()
        else:
            if r == 1:
                # Chebyshev nodes
                k = np.arange(8)
                x0_nodes = 0.5 * (1.0 + np.cos(np.pi * (2*k + 1) / 16.0))
                x0_nodes = np.sort(x0_nodes)
                x0_weights = np.ones(8) / 8.0
            elif r == 2:
                # Equally spaced
                x0_nodes = np.linspace(0.05, 0.95, 8)
                x0_weights = np.ones(8) / 8.0
            elif r == 3:
                # Gauss-Chebyshev type
                k = np.arange(8)
                x0_nodes = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / 8.0))
                x0_nodes = np.sort(x0_nodes)
                x0_weights = np.ones(8) / 8.0
            else:
                # Perturb best
                x0 = best_x.copy()
                x0 += np.random.normal(0, 0.01, 16)
                x0[:8] = np.sort(x0[:8])
                x0[:8] = np.clip(x0[:8], 1e-6, 1.0 - 1e-6)
                x0[8:] = np.clip(x0[8:], 1e-8, 1.0)
                x0[8:] /= np.sum(x0[8:])
            
            if r in [1, 2, 3]:
                x0 = np.concatenate([x0_nodes, x0_weights])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          callback=callback, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                          callback=callback, options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16})
        except Exception:
            pass
    
    # Ensure final result is valid
    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Sort nodes and re-associate weights (in case order was violated)
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = weights[sort_idx]
    
    # Check validity
    if (not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or 
        abs(np.sum(weights) - 1.0) > 1e-8 or nodes[0] <= 0 or nodes[-1] >= 1):
        # Fallback to Gauss-Legendre
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    else:
        # Normalize weights to sum exactly to 1
        weights = weights / np.sum(weights)
        # Ensure strict positivity
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
