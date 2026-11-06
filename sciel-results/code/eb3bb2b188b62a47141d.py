"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: must finish within 90s, aim to return well before
_DEADLINE = time.time() + 85.0  # 5s safety margin

def _E_of_z(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model in flat universe (matter + DE only)."""
    a = 1.0 / (1.0 + z)
    Omega_m = 0.3
    Omega_DE = 1.0 - Omega_m
    term_m = Omega_m / (a**3)
    term_de = Omega_DE * (a ** (3.0 * (w0 - wa))) * np.exp(3.0 * wa * (a - 1.0))
    E_sq = term_m + term_de
    return np.sqrt(np.maximum(E_sq, 1e-30))


def _high_accuracy_reference(z_test, w0, wa, n_pts=200):
    """High-accuracy reference using dense Gauss-Legendre quadrature."""
    # Use Gauss-Legendre with many points for high accuracy
    x, w = np.polynomial.legendre.leggauss(n_pts)
    # Map from [-1,1] to [0, z_test]
    z_pts = 0.5 * z_test * (x + 1.0)
    w_pts = 0.5 * z_test * w
    E_pts = _E_of_z(z_pts, w0, wa)
    integral = np.sum(w_pts / E_pts)
    return (1.0 + z_test) * integral


def _quadrature_value(nodes, weights, z_test, w0, wa):
    """Compute quadrature approximation of luminosity distance."""
    z_nodes = z_test * nodes
    E_nodes = _E_of_z(z_nodes, w0, wa)
    return (1.0 + z_test) * z_test * np.sum(weights / E_nodes)


def _relative_error(nodes, weights, z_test, w0, wa):
    """Compute relative error for a single z."""
    quad_val = _quadrature_value(nodes, weights, z_test, w0, wa)
    ref_val = _high_accuracy_reference(z_test, w0, wa)
    if abs(ref_val) < 1e-15:
        return 0.0
    return abs(quad_val - ref_val) / abs(ref_val)


def _worst_error(nodes, weights, z_values, models):
    """Compute worst relative error over z values and models."""
    max_err = 0.0
    for w0, wa in models:
        for z in z_values:
            err = _relative_error(nodes, weights, z, w0, wa)
            if err > max_err:
                max_err = err
    return max_err


def _objective(params):
    """Objective function to minimize: worst relative error."""
    if time.time() > _DEADLINE - 5.0:
        return 1e6
    
    nodes = params[:8]
    weights = params[8:]
    
    # Enforce constraints via penalties
    if nodes[0] < 1e-6 or nodes[-1] > 1.0 - 1e-6:
        return 1e6
    if np.any(np.diff(nodes) <= 0):
        return 1e6
    if np.any(weights <= 0):
        return 1e6
    if abs(np.sum(weights) - 1.0) > 1e-6:
        return 1e6
    
    # Quick evaluation with representative z and models
    z_values = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
    models = [(-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5), (-1.0, 0.0)]
    
    return _worst_error(nodes, weights, z_values, models)


def _refine_from(nodes, weights, max_iter=300):
    """Refine quadrature rule using Nelder-Mead."""
    params0 = np.concatenate([nodes, weights])
    
    # Use log-weights for positivity constraint
    # Actually, let's just use bounds and hope for the best
    result = minimize(_objective, params0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14,
                               'adaptive': True, 'initial_simplex': 1e-4})
    
    if result.fun < 1e5:  # Valid solution
        new_nodes = result.x[:8]
        new_weights = result.x[8:]
        # Normalize weights
        new_weights = new_weights / np.sum(new_weights)
        return new_nodes, new_weights
    return nodes, weights


def construct():
    t_start = time.time()
    
    # Gauss-Legendre nodes and weights as a strong starting point
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (gl_nodes + 1.0)
    weights_gl = 0.5 * gl_weights
    
    # Also try equally spaced
    nodes_eq = (np.arange(8) + 0.5) / 8.0
    weights_eq = np.full(8, 1.0/8.0)
    
    # Evaluate both starting points
    z_test = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0])
    models_test = [(-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5), (-1.0, 0.0), (0.0, -0.5)]
    
    err_gl = _worst_error(nodes_gl, weights_gl, z_test, models_test)
    err_eq = _worst_error(nodes_eq, weights_eq, z_test, models_test)
    
    if err_gl <= err_eq:
        best_nodes = nodes_gl.copy()
        best_weights = weights_gl.copy()
        best_err = err_gl
    else:
        best_nodes = nodes_eq.copy()
        best_weights = weights_eq.copy()
        best_err = err_eq
    
    # Try refinement from both starting points
    for init_nodes, init_weights in [(nodes_gl, weights_gl), (nodes_eq, weights_eq)]:
        if time.time() > _DEADLINE - 20.0:
            break
        
        try:
            refined_nodes, refined_weights = _refine_from(init_nodes.copy(), init_weights.copy())
            
            # Validate constraints
            if (np.all(np.diff(refined_nodes) > 0) and
                np.all(refined_nodes > 1e-6) and
                np.all(refined_nodes < 1.0 - 1e-6) and
                np.all(refined_weights > 0) and
                abs(np.sum(refined_weights) - 1.0) < 1e-10):
                
                refined_weights = refined_weights / np.sum(refined_weights)
                err = _worst_error(refined_nodes, refined_weights, z_test, models_test)
                
                if err < best_err:
                    best_err = err
                    best_nodes = refined_nodes
                    best_weights = refined_weights
        except Exception:
            pass
    
    # Second round of refinement if time allows
    if time.time() < _DEADLINE - 25.0:
        try:
            refined_nodes, refined_weights = _refine_from(best_nodes.copy(), best_weights.copy(), max_iter=200)
            if (np.all(np.diff(refined_nodes) > 0) and
                np.all(refined_nodes > 1e-6) and
                np.all(refined_nodes < 1.0 - 1e-6) and
                np.all(refined_weights > 0)):
                refined_weights = refined_weights / np.sum(refined_weights)
                err = _worst_error(refined_nodes, refined_weights, z_test, models_test)
                if err < best_err:
                    best_err = err
                    best_nodes = refined_nodes
                    best_weights = refined_weights
        except Exception:
            pass
    
    # Final validation with broader test set
    if time.time() < _DEADLINE - 10.0:
        z_final = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
        models_final = [(-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5), (0.0, -0.5), (-1.0, 0.0), (0.5, 0.0)]
        final_err = _worst_error(best_nodes, best_weights, z_final, models_final)
        
        # If final error is much worse than expected, fall back to GL
        if final_err > best_err * 2.0:
            best_nodes = nodes_gl.copy()
            best_weights = weights_gl.copy()
    
    # Ensure final constraints
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-10)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure strictly increasing nodes in (0,1)
    for i in range(7):
        if best_nodes[i+1] <= best_nodes[i]:
            best_nodes[i+1] = best_nodes[i] + 1e-10
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
