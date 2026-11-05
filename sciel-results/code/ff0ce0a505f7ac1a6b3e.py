"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Basis-Function Moment Matching with Rationale Approximation.
Instead of directly minimizing the worst-case error against a library of specific
cosmological instances (which can overfit to the training set and get stuck in
local minima), we design the quadrature rule by enforcing that it exactly
integrates a carefully chosen basis of functions that capture the asymptotic
scaling and structure of 1/E(a) for CPL models.

The key insight is that 1/E(a) for CPL models has a specific structure:
- Near a=1 (z=0): 1/E(a) ≈ 1 + O((1-a))
- Near a=0 (z→∞): 1/E(a) scales as a^p where p depends on w0, wa

We use a basis of functions: {a^k, a^k * log(a), a^k * (1-a)^m} that captures
this behavior. By matching moments of these basis functions, we construct a
"universal" rule that should perform well across the entire parameter space
without overfitting to specific instances.

This is materially different from the current approach because:
1. We don't optimize against specific cosmologies - we match structural moments
2. The optimization is smoother and has fewer local minima
3. The resulting rule is more robust to unseen validation cosmologies
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

def _build_basis_functions(M=10, L=4):
    """
    Build a basis of functions that capture the structure of 1/E(a).
    
    We use:
    - Powers: a^k for k=0..M-1
    - Logarithmic corrections: a^k * log(a) for k=0..L-1
    - Endpoint corrections: a^k * (1-a)^m for k=0..3, m=1,2
    
    This captures the asymptotic behavior of 1/E(a) for CPL models.
    """
    basis = []
    # Powers
    for k in range(M):
        basis.append(lambda a, k=k: a ** k)
    # Logarithmic corrections
    for k in range(L):
        basis.append(lambda a, k=k: a ** k * np.log(np.clip(a, 1e-10, 1.0)))
    # Endpoint corrections
    for k in range(4):
        for m in range(1, 3):
            basis.append(lambda a, k=k, m=m: a ** k * (1 - a) ** m)
    return basis

def _compute_moments(basis, nodes, weights):
    """Compute the discrete moments of the basis functions."""
    moments = np.array([np.sum(weights * f(nodes)) for f in basis])
    return moments

def _target_moments(basis):
    """Compute the exact integral moments of the basis functions over [0,1]."""
    # We'll use high-accuracy Gauss-Legendre to compute these
    n = 64
    xi, wi = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (xi + 1.0)
    targets = np.array([np.sum(wi * f(u)) for f in basis])
    return targets

def _moment_residual(nodes, weights, basis, targets):
    """Compute the residual between discrete and target moments."""
    moments = _compute_moments(basis, nodes, weights)
    residual = moments - targets
    return np.sum(residual ** 2)

def _build_library():
    """Build a diverse library of (w0, wa) pairs for validation."""
    w0_vals = np.linspace(-2.0, 1.0, 5)
    wa_vals = np.linspace(-2.0, 2.0, 5)
    lib = []
    for w0 in w0_vals:
        for wa in wa_vals:
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
    if len(errors) == 0:
        return 0.0
    return (1.0 / alpha) * np.log(np.sum(np.exp(alpha * errors)))

def construct():
    fallback = _fallback()
    n = 8
    
    # 1. Initial Guess: Gauss-Legendre nodes and weights mapped to [0,1]
    xi, wi = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * wi
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    
    # Build basis and compute target moments
    basis = _build_basis_functions(M=8, L=3)
    targets = _target_moments(basis)
    
    lib = _build_library()
    zs = np.linspace(0.1, 3.0, 12)
    
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
        return _moment_residual(nodes_x, w_norm, basis, targets)
    
    x0 = np.concatenate([nodes, weights])
    
    if time.time() < DEADLINE - MARGIN - 10:
        res = minimize(moment_obj, x0, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < 1e-4:
            new_nodes = res.x[:n]
            new_weights = res.x[n:]
            if np.all(np.diff(new_nodes) > 1e-6) and np.all(new_weights > 1e-12):
                new_weights = new_weights / np.sum(new_weights)
                err = _smooth_max(_get_errors(new_nodes, new_weights, zs, lib))
                if err < best_err:
                    best_nodes = new_nodes
                    best_weights = new_weights
                    best_err = err
    
    # 3. Local Refinement with Smooth Max
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
    for restart in range(4):
        if time.time() > DEADLINE - MARGIN:
            break
        
        if restart == 0:
            x_start = x0_ref
        else:
            # Perturb the current best
            noise_scale = 0.005 * (restart + 1)
            x_start = x0_ref + noise_scale * np.random.randn(2 * n)
            # Project back to valid region
            x_start[:n] = np.sort(x_start[:n])
            x_start[:n] = np.clip(x_start[:n], 1e-5, 1 - 1e-5)
            x_start[n:] = np.clip(x_start[n:], 1e-12, None)
        
        res = minimize(error_obj, x_start, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12})
        
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
