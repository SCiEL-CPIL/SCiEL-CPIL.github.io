import numpy as np
import time
import scipy.optimize as opt
from scipy.integrate import quad as scipy_quad

def _cpl_E(z, w0, w1):
    """CPL E(z) = ((1+z)^3 * (1 + w0 + w1*z))^(1/3) for flat universe."""
    return ((1.0 + z)**3 * (1.0 + w0 + w1 * z))**(1.0 / 3.0)

def _integrand(z, w0, w1):
    """1/E(z) for CPL model."""
    return 1.0 / _cpl_E(z, w0, w1)

def _exact_luminosity(z_max, w0, w1):
    """Exact integral of z/E(z) from 0 to z_max using adaptive quadrature."""
    f = lambda zz: zz / _cpl_E(zz, w0, w1)
    result, _ = scipy_quad(f, 0.0, z_max, limit=200)
    return result

def _quadrature_luminosity(z_max, nodes, weights, w0, w1):
    """Quadrature approximation of integral of z/E(z) from 0 to z_max."""
    z_eval = z_max * nodes
    vals = z_eval / _cpl_E(z_eval, w0, w1)
    return np.dot(weights, vals)

def _compute_worst_error(nodes, weights, z_max, models):
    """Compute worst relative error across all models."""
    max_err = 0.0
    for (w0, w1) in models:
        exact = _exact_luminosity(z_max, w0, w1)
        quad_val = _quadrature_luminosity(z_max, nodes, weights, w0, w1)
        if abs(exact) < 1e-15:
            err = abs(quad_val - exact)
        else:
            err = abs(quad_val - exact) / abs(exact)
        if err > max_err:
            max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Representative cosmological models (w0, w1) covering a wide range
    # LambdaCDM, phantom, quintessence, various CPL parameters
    models = [
        (-1.0, 0.0),      # LambdaCDM
        (-1.0, 0.5),      # CPL with positive w1
        (-1.0, -0.5),     # CPL with negative w1
        (-0.9, 0.0),      # Slightly non-cosmological constant
        (-0.9, 0.3),      # Quintessence-like
        (-0.9, -0.3),     # Phantom-like
        (-0.8, 0.0),      # More matter-like
        (-0.8, 0.5),      # Strong CPL
        (-0.8, -0.5),     # Strong phantom CPL
        (-1.1, 0.0),      # Phantom constant
        (-1.1, 0.3),      # Phantom with CPL
        (-1.1, -0.3),     # Phantom with negative CPL
        (-0.7, 0.0),      # More matter-dominated
        (-0.7, 0.4),      # Matter-like with CPL
        (-0.7, -0.4),     # Matter-like with phantom CPL
        (-1.0, 0.8),      # Very strong CPL
        (-1.0, -0.8),     # Very strong phantom CPL
        (-0.5, 0.0),      # Very matter-dominated
        (-0.5, 0.5),      # Matter with CPL
        (-0.5, -0.5),     # Matter with phantom
    ]
    
    z_max = 1.0  # Standard validation redshift
    
    # Start with Gauss-Legendre as initial guess
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = (x_gl + 1.0) / 2.0
    weights_init = w_gl / 2.0
    weights_init = weights_init / np.sum(weights_init)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_error = _compute_worst_error(best_nodes, best_weights, z_max, models)
    
    # Use log-parameterized weights to ensure positivity
    # And optimize nodes and log-weights directly
    def _objective(params):
        if time.time() > deadline:
            return np.inf
        n = params[:8]
        log_w = params[8:]
        w = np.exp(log_w)
        w = w / np.sum(w)  # Normalize to sum to 1
        
        # Check node constraints
        if not np.all(n > 1e-10) or not np.all(n < 1.0 - 1e-10):
            return np.inf
        if not np.all(np.diff(n) > 1e-10):
            return np.inf
        if not np.all(w > 1e-15):
            return np.inf
            
        return _compute_worst_error(n, w, z_max, models)
    
    # Initial parameter vector
    x0 = np.concatenate([nodes_init, np.log(weights_init)])
    
    # Bounds: nodes in (1e-8, 1-1e-8), log-weights in (-30, 0)
    bounds = [(1e-8, 1.0 - 1e-8)] * 8 + [(-30.0, 0.0)] * 8
    
    # Try multiple optimization approaches
    attempts = []
    
    # Attempt 1: SLSQP
    try:
        res = opt.minimize(
            _objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
        )
        if res.success and time.time() < deadline:
            n_opt = res.x[:8]
            w_opt = np.exp(res.x[8:])
            w_opt = w_opt / np.sum(w_opt)
            if np.all(n_opt > 1e-10) and np.all(n_opt < 1.0 - 1e-10) and np.all(np.diff(n_opt) > 1e-10) and np.all(w_opt > 1e-15):
                err = _compute_worst_error(n_opt, w_opt, z_max, models)
                attempts.append((err, n_opt, w_opt))
    except Exception:
        pass
    
    # Attempt 2: Trust-constr
    if time.time() < deadline:
        try:
            res = opt.minimize(
                _objective,
                x0,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            if time.time() < deadline:
                n_opt = res.x[:8]
                w_opt = np.exp(res.x[8:])
                w_opt = w_opt / np.sum(w_opt)
                if np.all(n_opt > 1e-10) and np.all(n_opt < 1.0 - 1e-10) and np.all(np.diff(n_opt) > 1e-10) and np.all(w_opt > 1e-15):
                    err = _compute_worst_error(n_opt, w_opt, z_max, models)
                    attempts.append((err, n_opt, w_opt))
        except Exception:
            pass
    
    # Attempt 3: L-BFGS-B with penalty for constraints
    if time.time() < deadline:
        def _penalty_objective(params):
            n = params[:8]
            log_w = params[8:]
            w = np.exp(log_w)
            w = w / np.sum(w)
            
            penalty = 0.0
            # Penalize nodes out of bounds
            penalty += np.sum(np.maximum(0, 1e-8 - n)) * 1e6
            penalty += np.sum(np.maximum(0, n - (1.0 - 1e-8))) * 1e6
            # Penalize non-increasing nodes
            diffs = np.diff(n)
            penalty += np.sum(np.maximum(0, 1e-10 - diffs)) * 1e6
            # Penalize very small weights
            penalty += np.sum(np.maximum(0, 1e-15 - w)) * 1e6
            
            return _compute_worst_error(n, w, z_max, models) + penalty
        
        try:
            res = opt.minimize(
                _penalty_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if time.time() < deadline:
                n_opt = res.x[:8]
                w_opt = np.exp(res.x[8:])
                w_opt = w_opt / np.sum(w_opt)
                if np.all(n_opt > 1e-10) and np.all(n_opt < 1.0 - 1e-10) and np.all(np.diff(n_opt) > 1e-10) and np.all(w_opt > 1e-15):
                    err = _compute_worst_error(n_opt, w_opt, z_max, models)
                    attempts.append((err, n_opt, w_opt))
        except Exception:
            pass
    
    # Select the best attempt
    if attempts:
        attempts.sort(key=lambda x: x[0])
        best_err, best_nodes, best_weights = attempts[0]
    
    # Final refinement: local Nelder-Mead on the best solution
    if time.time() < deadline and best_error < np.inf:
        try:
            def _final_obj(params):
                n = params[:8]
                log_w = params[8:]
                w = np.exp(log_w)
                w = w / np.sum(w)
                if not np.all(n > 1e-10) or not np.all(n < 1.0 - 1e-10):
                    return np.inf
                if not np.all(np.diff(n) > 1e-10):
                    return np.inf
                return _compute_worst_error(n, w, z_max, models)
            
            res = opt.minimize(
                _final_obj,
                np.concatenate([best_nodes, np.log(best_weights)]),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if time.time() < deadline:
                n_opt = res.x[:8]
                w_opt = np.exp(res.x[8:])
                w_opt = w_opt / np.sum(w_opt)
                if np.all(n_opt > 1e-10) and np.all(n_opt < 1.0 - 1e-10) and np.all(np.diff(n_opt) > 1e-10) and np.all(w_opt > 1e-15):
                    err = _compute_worst_error(n_opt, w_opt, z_max, models)
                    if err < best_error:
                        best_nodes = n_opt
                        best_weights = w_opt
                        best_error = err
        except Exception:
            pass
    
    # Final cleanup: ensure strict constraints
    eps = 1e-12
    best_nodes = np.clip(best_nodes, eps, 1.0 - eps)
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
