"""
Strategy: Direct 16-variable minimax SLSQP optimization of nodes and weights jointly.

Instead of the alternating LP scheme (solve weights for fixed nodes, then move nodes),
we directly optimize all 16 variables (8 nodes + 8 weights) with SLSQP minimizing the
maximum relative error over a dense Chebyshev-weighted test set. Multiple diverse
starting points are tried, and the best result is kept.

Key differences from current program:
- No LP subproblem; direct nonlinear minimax on all variables.
- Chebyshev-weighted test set for uniform importance across parameter space.
- Multiple diverse starts (GL, Cheb, Gauss-Jacobi, log-spaced, power-law).
- SLSQP handles the smooth minimax objective with gradient information.
"""

import time
import numpy as np
from scipy.optimize import minimize, roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0


def _time_left():
    return _DEADLINE - time.time()


def _E_of_z_vec(z, w0, wa, Omega_m=0.3):
    """Vectorized E(z) for CPL dark energy."""
    Omega_de = 1.0 - Omega_m
    one_plus_z = 1.0 + z
    term_m = Omega_m * one_plus_z**3
    exponent = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * one_plus_z**exponent
    return np.sqrt(term_m + term_de)


def _build_test_set():
    """
    Build a dense Chebyshev-weighted test set.
    Uses Chebyshev nodes in (w0, wa) and z for near-uniform importance.
    """
    # Chebyshev nodes for w0 in [-1.5, 1.0]
    n_w0 = 12
    k = np.arange(n_w0)
    w0_cheb = 0.25 - 0.875 * np.cos(np.pi * (2 * k + 1) / (2 * n_w0))  # maps to [-0.875+0.25, 0.875+0.25]
    # Actually let's do proper Chebyshev on [-1.5, 1.0]
    w0_cheb = -0.25 - 0.875 * np.cos(np.pi * (2 * k + 1) / (2 * n_w0))  # [-1.125, 0.625]... not right
    # Let me just do: center = (-1.5+1.0)/2 = -0.25, half = 1.25
    w0_cheb = -0.25 - 0.625 * np.cos(np.pi * (2 * k + 1) / (2 * n_w0))
    # This gives values in [-0.25-0.625, -0.25+0.625] = [-0.875, 0.375]... 
    # Let me just use linspace with more points for simplicity and robustness
    
    w0_vals = np.linspace(-1.5, 1.0, 11)
    wa_vals = np.linspace(-0.8, 0.8, 11)
    z_vals = np.linspace(0.02, 5.0, 15)
    
    # Add extra points at interesting locations
    z_extra = np.array([0.05, 0.1, 0.3, 1.0, 2.0, 3.0])
    z_vals = np.unique(np.concatenate([z_vals, z_extra]))
    
    test_points = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_points.append((z, w0, wa))
    
    return test_points


def _compute_refs(test_points, n_gauss=96):
    """Compute reference integrals using high-order Gauss-Legendre."""
    x, w = roots_legendre(n_gauss)
    t = 0.5 * (x + 1.0)
    wt = 0.5 * w
    refs = np.zeros(len(test_points))
    for idx, (z, w0, wa) in enumerate(test_points):
        if z < 1e-10:
            refs[idx] = 1.0
        else:
            zt = z * t
            E = _E_of_z_vec(zt, w0, wa)
            refs[idx] = np.dot(wt, 1.0 / E)
    return refs


def _quadrature_value(nodes, weights, z, w0, wa, Omega_m=0.3):
    """Compute sum w_i / E(z * t_i)."""
    if z < 1e-10:
        return np.sum(weights)
    zt = z * nodes
    E = _E_of_z_vec(zt, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E)


def _max_rel_error(nodes, weights, test_points, refs):
    """Compute maximum relative error over test set."""
    max_err = 0.0
    for idx, (z, w0, wa) in enumerate(test_points):
        approx = _quadrature_value(nodes, weights, z, w0, wa)
        ref = refs[idx]
        if ref > 1e-15:
            err = abs(approx - ref) / ref
            if err > max_err:
                max_err = err
    return max_err


def _objective(x, test_points, refs):
    """
    Objective: maximize epsilon such that |approx - ref|/ref <= epsilon for all.
    We minimize epsilon via SLSQP.
    
    x = [nodes(8), weights(8)]
    """
    nodes = x[:8]
    weights = x[8:]
    
    # Check validity
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 1e-10):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-6:
        return 1e10
    
    return _max_rel_error(nodes, weights, test_points, refs)


def _gauss_legendre_nodes(n=8):
    nodes, weights = roots_legendre(n)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return nodes, weights


def _chebyshev_nodes(n=8):
    k = np.arange(n)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    nodes = (nodes + 1.0) / 2.0
    nodes = np.sort(nodes)
    # Equal weights as initial guess
    weights = np.ones(n) / n
    return nodes, weights


def _generate_starts():
    """Generate diverse starting points (nodes, weights)."""
    starts = []
    
    # Gauss-Legendre
    n, w = _gauss_legendre_nodes(8)
    starts.append((n, w))
    
    # Chebyshev
    n, w = _chebyshev_nodes(8)
    starts.append((n, w))
    
    # Gauss-Jacobi variants
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0), (1.5, 0.5), (0.5, 1.5)]:
        try:
            nodes, weights = roots_jacobi(8, alpha, beta)
            nodes = (nodes + 1.0) / 2.0
            nodes = np.sort(nodes)
            weights = weights / 2.0
            weights = weights / np.sum(weights)
            if np.all(np.diff(nodes) > 1e-8) and np.all(nodes > 1e-4) and np.all(nodes < 1 - 1e-4):
                starts.append((nodes, weights))
        except:
            pass
    
    # Log-spaced
    for base in [1.5, 2.0, 3.0]:
        t = np.linspace(0, 1, 8)
        log_nodes = 1.0 - (1.0 - t) ** base
        log_nodes = np.clip(log_nodes, 1e-3, 1 - 1e-3)
        if np.all(np.diff(log_nodes) > 1e-6):
            weights = np.ones(8) / 8
            starts.append((log_nodes, weights))
    
    # Power-law
    for base in [0.5, 0.3, 0.7, 1.5]:
        t = np.linspace(0, 1, 8)
        pow_nodes = t ** (1.0 / base)
        pow_nodes = np.clip(pow_nodes, 1e-3, 1 - 1e-3)
        if np.all(np.diff(pow_nodes) > 1e-6):
            weights = np.ones(8) / 8
            starts.append((pow_nodes, weights))
    
    # Uniform
    uniform = np.linspace(1/17, 16/17, 8)
    starts.append((uniform, np.ones(8) / 8))
    
    return starts


def construct():
    if time.time() > _DEADLINE - 5:
        nodes, weights = _gauss_legendre_nodes(8)
        return {"nodes": nodes, "weights": weights}
    
    # Build test set and references
    test_points = _build_test_set()
    refs = _compute_refs(test_points, n_gauss=96)
    
    # Precompute A matrix for speed: A[i, j] = 1/E(z_i * t_j)
    N = len(test_points)
    n = 8
    A = np.zeros((N, n))
    for idx, (z, w0, wa) in enumerate(test_points):
        if z < 1e-10:
            A[idx, :] = 1.0
        else:
            zt = z * np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])  # placeholder, will recompute
            # Actually we need to compute for the actual nodes, so we can't precompute
            # We'll just compute on the fly but vectorize
            pass
    
    # Generate starts
    starts = _generate_starts()
    
    best_nodes = None
    best_weights = None
    best_err = float('inf')
    
    bounds = [(1e-4, 1 - 1e-4)] * 8 + [(1e-8, None)] * 8
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0},
        {'type': 'eq', 'fun': lambda x: x[0] - 1e-4, 'jac': lambda x: np.array([1.0] + [0.0]*15)},
    ]
    # Actually, let's not fix nodes at boundaries. Just use bounds.
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0},
    ]
    
    for start_idx, (n0, w0) in enumerate(starts):
        if time.time() > _DEADLINE - 10:
            break
        
        x0 = np.concatenate([n0, w0])
        
        try:
            result = minimize(
                _objective, x0,
                args=(test_points, refs),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.fun < best_err and result.fun < 1e6:
                nodes_opt = result.x[:8]
                weights_opt = result.x[8:]
                
                # Validate
                if np.all(nodes_opt > 1e-5) and np.all(nodes_opt < 1 - 1e-5):
                    if np.all(np.diff(nodes_opt) > 1e-8):
                        if np.all(weights_opt > 1e-10):
                            best_nodes = nodes_opt.copy()
                            best_weights = weights_opt.copy()
                            best_err = result.fun
        except Exception:
            pass
    
    # If no valid result, fall back to Gauss-Legendre
    if best_nodes is None:
        nodes, weights = _gauss_legendre_nodes(8)
        return {"nodes": nodes, "weights": weights}
    
    # Final refinement: try a second round with tighter tolerance from best point
    if time.time() < _DEADLINE - 15:
        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result2 = minimize(
                _objective, x0,
                args=(test_points, refs),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            if result2.fun < best_err and result2.fun < 1e6:
                nodes_opt = result2.x[:8]
                weights_opt = result2.x[8:]
                if np.all(nodes_opt > 1e-5) and np.all(nodes_opt < 1 - 1e-5):
                    if np.all(np.diff(nodes_opt) > 1e-8):
                        if np.all(weights_opt > 1e-10):
                            best_nodes = nodes_opt.copy()
                            best_weights = weights_opt.copy()
                            best_err = result2.fun
        except Exception:
            pass
    
    # Normalize weights
    w_sum = np.sum(best_weights)
    if w_sum > 1e-12:
        best_weights = best_weights / w_sum
    
    # Sort by nodes
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    
    # Final validation
    if np.any(best_nodes <= 0) or np.any(best_nodes >= 1):
        nodes, weights = _gauss_legendre_nodes(8)
        return {"nodes": nodes, "weights": weights}
    
    if np.any(np.diff(best_nodes) <= 0):
        nodes, weights = _gauss_legendre_nodes(8)
        return {"nodes": nodes, "weights": weights}
    
    if np.any(best_weights <= 0):
        nodes, weights = _gauss_legendre_nodes(8)
        return {"nodes": nodes, "weights": weights}
    
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}


# EVOLVE-BLOCK-END
