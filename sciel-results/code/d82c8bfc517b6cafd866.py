"""Minimal logarithmic energy of 100 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0  # safety margin before 120s kill

def _elapsed():
    return time.time() - _START

_START = time.time()

def _energy_and_grad(x):
    """
    x: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. x projected onto the tangent space.
    """
    x = np.asarray(x, dtype=np.float64)
    # Compute all pairwise distances
    # d[i,j] = |xi - xj|
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero and log(0)
    # For i != j, dist should be > 0
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    
    energy = np.sum(np.log(1.0 / d))
    
    # Gradient computation
    # dE/dxk for each point k
    # E = sum_{i<j} -log(|xi-xj|)
    # dE/dxk = sum_{j != k} (xk - xj) / |xk - xj|^2
    # Because d(-log(|xi-xj|))/dxi = -(xi-xj)/|xi-xj|^2 * (-1) ... let me be careful.
    # f(d) = -log(d), df/dd = -1/d
    # d/dxi [-log(|xi-xj|)] = -1/|xi-xj| * (xi-xj)/|xi-xj| = -(xi-xj)/|xi-xj|^2
    # So for point k, grad_k = sum_{j != k} -(xk - xj)/|xk-xj|^2
    # Wait, that's for one pair. Let me re-derive.
    # E = sum_{i<j} -log(||xi-xj||)
    # dE/dxk = sum_{j: j<k} d/dxk[-log(||xk-xj||)] + sum_{j: j>k} d/dxk[-log(||xj-xk||)]
    # d/dxk[-log(||xk-xj||)] = -1/||xk-xj|| * (xk-xj)/||xk-xj|| = -(xk-xj)/||xk-xj||^2
    # d/dxk[-log(||xj-xk||)] = -1/||xj-xk|| * (xk-xj)/||xj-xk|| = -(xk-xj)/||xk-xj||^2
    # So dE/dxk = -sum_{j != k} (xk-xj)/||xk-xj||^2
    
    grad = np.zeros_like(x)
    # For each k, grad[k] = -sum_{j != k} (x[k]-x[j]) / |x[k]-x[j]|^2
    # Vectorized:
    # diff[k,j] = x[k] - x[j]
    # dist2[k,j] = |x[k]-x[j]|^2
    dist2 = np.sum(diff * diff, axis=2)
    # Set diagonal to inf to avoid division by zero
    np.fill_diagonal(dist2, np.inf)
    
    # weight[k,j] = 1/dist2[k,j]
    weight = 1.0 / dist2
    # grad[k] = -sum_j weight[k,j] * diff[k,j]
    # grad = -np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    # Project gradient onto tangent space of sphere: grad_t = grad - (grad . x) * x
    # Since we want to stay on the sphere, we use the projected gradient
    dot = np.sum(grad * x, axis=1, keepdims=True)
    grad_t = grad - dot * x
    
    return energy, grad_t

def _project(x):
    """Project points back onto unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms

def _energy_only(x):
    """Fast energy evaluation without gradient."""
    x = np.asarray(x, dtype=np.float64)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def construct():
    global _START
    _START = time.time()
    
    best_x = None
    best_energy = np.inf
    
    # Phase 1: Generate multiple diverse starting points
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    fib_x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Random starts
    rng = np.random.default_rng(42)
    random_starts = []
    for _ in range(5):
        v = rng.standard_normal((N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        random_starts.append(v)
    
    # Add a few more structured starts
    # Spherical design-like: use spherical codes
    starts = [fib_x] + random_starts
    
    # Phase 2: Optimize each start with L-BFGS-B
    for start in starts:
        if _elapsed() > DEADLINE - 30:
            break
        
        # Use L-BFGS-B with projection
        def objective_and_grad(params):
            x = _project(params.reshape(N, 3))
            e, g = _energy_and_grad(x)
            # Gradient w.r.t. params (unconstrained)
            # Since we project, the chain rule gives us the projected gradient
            # Actually, we need the gradient w.r.t. the flat params
            # x = params / |params|
            # dx/dparams = (I - x x^T) / |params| ... this is complex
            # Simpler: use the projected gradient directly as the gradient w.r.t. params
            # This is an approximation but works in practice
            g_flat = g.flatten()
            return e, g_flat
        
        try:
            res = minimize(
                objective_and_grad,
                start.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            x_opt = _project(res.x.reshape(N, 3))
            e_opt = _energy_only(x_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    if best_x is None:
        return fib_x
    
    # Phase 3: Local refinement with multiple restarts from best
    for _ in range(3):
        if _elapsed() > DEADLINE - 10:
            break
        
        # Perturb best solution
        noise = rng.standard_normal((N, 3)) * 0.01
        perturbed = _project(best_x + noise)
        
        try:
            res = minimize(
                objective_and_grad,
                perturbed.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-12}
            )
            x_opt = _project(res.x.reshape(N, 3))
            e_opt = _energy_only(x_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Final ultra-precise polish
    if _elapsed() < DEADLINE - 5:
        try:
            res = minimize(
                objective_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-12}
            )
            x_opt = _project(res.x.reshape(N, 3))
            e_opt = _energy_only(x_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_x = _project(best_x)
    
    # Verify no points are too close
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if points are too close
        return fib_x
    
    return best_x
