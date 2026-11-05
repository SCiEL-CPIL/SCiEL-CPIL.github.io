"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Moment Matching with Robust Local Search.
Instead of spectral approximation (which assumes smoothness and may miss the
worst-case behavior) or direct minimax (which is non-smooth and hard to optimize),
we use moment matching: we design the quadrature rule to exactly integrate a
set of carefully chosen test functions that span the space of possible integrands
1/E(a) for CPL models. We then perform a robust local search using a
differentiable surrogate for the worst-case error.

Key differences from the current program:
1. We use moment matching with a broader set of test functions (not just Chebyshev)
   that better capture the structure of 1/E(a) for CPL models.
2. We use a differentiable surrogate for the worst-case error based on a smooth
   max approximation (log-sum-exp) instead of the non-smooth max.
3. We use a more robust local search with multiple restarts and a wider
   cosmology library that includes more extreme models.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

START = time.time()
DEADLINE = 85.0
MARGIN = 5.0

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, w0, wa):
    """CPL E(a) for flat universe."""
    a = np.clip(a, 1e-8, 1.0)
    Omega_m = 0.3
    Omega_L0 = 0.7
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    pow_term = a ** (-3.0 * (1.0 + w0 + wa))
    Omega_L = Omega_L0 * pow_term * exp_term
    E2 = a ** (-3.0) + Omega_L
    return np.sqrt(E2)

def _integrate(z, nodes, weights, Efun):
    """Compute the quadrature approximation."""
    a = 1.0 / (1.0 + z * nodes)
    E = Efun(a)
    return (1.0 + z) * z * np.sum(weights / E)

def _ref_integrate(z, Efun):
    """High-accuracy reference integration using Gauss-Legendre."""
    n = 64
    xi, wi = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (xi + 1.0)
    a = 1.0 / (1.0 + z * u)
    E = Efun(a)
    val = z * np.sum(wi / E)
    return (1.0 + z) * val

def _build_library():
    """Build a diverse library of (w0, wa) pairs for validation."""
    w0_vals = np.linspace(-3.0, 1.0, 7)
    wa_vals = np.linspace(-3.0, 3.0, 7)
    lib = []
    for w0 in w0_vals:
        for wa in wa_vals:
            if abs(w0 + 1.0) < 0.05 and abs(wa) < 0.05:
                continue
            lib.append((w0, wa))
    return lib

def _get_errors(nodes, weights, zs, lib):
    """Compute all relative errors for the given nodes/weights."""
    errors = []
    for w0, wa in lib:
        Efun = lambda a, w0=w0, wa=wa: _E_cpl(a, w0, wa)
        for z in zs:
            app = _integrate(z, nodes, weights, Efun)
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            err = abs(app - ref) / ref
            errors.append(err)
    return np.array(errors)

def _smooth_max(errors, alpha=50.0):
    """Smooth approximation to max using log-sum-exp."""
    return (1.0 / alpha) * np.log(np.sum(np.exp(alpha * errors)))

def _moment_residual(nodes, weights, M=12):
    """
    Compute the residual of the quadrature rule against monomials.
    We want sum_i w_i u_i^k to approximate integral_0^1 u^k du = 1/(k+1).
    """
    n = len(nodes)
    u = nodes
    target = np.array([1.0 / (k + 1) for k in range(M)])
    quad_vals = np.array([np.sum(weights * u**k) for k in range(M)])
    residual = quad_vals - target
    return np.sum(residual**2)

def construct():
    fallback = _fallback()
    n = 8
    
    # 1. Initial Guess: Gauss-Legendre nodes and weights mapped to [0,1]
    xi, wi = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * wi
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    
    lib = _build_library()
    zs = np.linspace(0.1, 3.0, 15)
    
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = float('inf')
    
    # 2. Moment Matching Optimization
    def moment_obj(x):
        nodes_x = x[:n]
        weights_x = x[n:]
        if nodes_x[0] < 1e-6 or nodes_x[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes_x) < 1e-6):
            return 1e10
        if np.any(weights_x < 1e-12):
            return 1e10
        w_norm = weights_x / np.sum(weights_x)
        return _moment_residual(nodes_x, w_norm, M=12)
    
    x0 = np.concatenate([nodes, weights])
    
    if time.time() < DEADLINE - MARGIN - 15:
        res = minimize(moment_obj, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < 1e-6:
            new_nodes = res.x[:n]
            new_weights = res.x[n:]
            if np.all(np.diff(new_nodes) > 1e-6) and np.all(new_weights > 1e-12):
                new_weights = new_weights / np.sum(new_weights)
                err = _smooth_max(_get_errors(new_nodes, new_weights, zs, lib))
                if err < best_err:
                    best_nodes = new_nodes
                    best_weights = new_weights
                    best_err = err
    
    # 3. Robust Local Search with Smooth Max
    def error_obj(x):
        nodes_x = x[:n]
        weights_x = x[n:]
        if nodes_x[0] < 1e-6 or nodes_x[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes_x) < 1e-6):
            return 1e10
        if np.any(weights_x < 1e-12):
            return 1e10
        weights_norm = weights_x / np.sum(weights_x)
        errors = _get_errors(nodes_x, weights_norm, zs, lib)
        return _smooth_max(errors, alpha=50.0)
    
    x0_ref = np.concatenate([best_nodes, best_weights])
    
    # Multiple restarts with perturbations
    for restart in range(3):
        if time.time() > DEADLINE - MARGIN:
            break
        
        if restart == 0:
            x_start = x0_ref
        else:
            # Perturb the current best
            noise_scale = 0.01 * (restart + 1)
            x_start = x0_ref + noise_scale * np.random.randn(2 * n)
            # Project back to valid region
            x_start[:n] = np.sort(x_start[:n])
            x_start[:n] = np.clip(x_start[:n], 1e-5, 1 - 1e-5)
            x_start[n:] = np.clip(x_start[n:], 1e-12, None)
        
        res = minimize(error_obj, x_start, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res.fun < best_err:
            new_nodes = res.x[:n]
            new_weights = res.x[n:]
            if np.all(np.diff(new_nodes) > 1e-6) and np.all(new_weights > 1e-12):
                new_weights = new_weights / np.sum(new_weights)
                best_nodes = new_nodes
                best_weights = new_weights
                best_err = res.fun
    
    # Final validation and return
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
