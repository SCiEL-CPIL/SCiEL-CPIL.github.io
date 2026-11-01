"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
MAX_TIME = 115.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START

def _energy_and_grad(x):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    x: (N, 3) array of points on unit sphere
    Returns: (energy, grad) where grad is (N, 3)
    """
    # Compute pairwise distances
    # dist[i,j] = |x_i - x_j|
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))  # (N, N)
    
    # Avoid log(0) for diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy = -sum_{i<j} log(dist[i,j])
    # Only use upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i [-log(|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: we need to be careful with the gradient
    # d/dx_i log(1/d) = d/dx_i (-log(d)) = -(x_i - x_j)/d^2 ... wait
    # Let's recompute: E = sum_{i<j} -log(d_ij)
    # dE/dx_i = sum_{j != i} d/dx_i [-log(d_ij)]
    # d/dx_i log(d_ij) = (x_i - x_j) / d_ij^2 (for i < j)
    # But for j < i, d/dx_i log(d_ji) = (x_i - x_j) / d_ji^2 as well
    # So dE/dx_i = -sum_{j != i} (x_i - x_j) / d_ij^2
    
    grad = np.zeros_like(x)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d < 1e-15:
                continue
            grad[i] -= (x[i] - x[j]) / (d * d)
    
    return energy, grad


def _energy_fast(x):
    """Fast energy computation using vectorized operations."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


def _project_to_sphere(x):
    """Project points to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _golden_spiral(offset=0.0):
    """Generate golden-angle Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(x0, time_limit, rng, label=""):
    """Run Riemannian optimization from a starting configuration.
    
    Uses scipy minimize with SLSQP or L-BFGS-B on the sphere constraint,
    followed by a custom Riemannian gradient descent polish.
    """
    start_time = time.time()
    x0 = _project_to_sphere(x0.copy())
    
    # First try scipy minimize with SLSQP
    try:
        def fun(x_flat):
            x = x_flat.reshape(N, 3)
            e, _ = _energy_and_grad(x)
            return e
        
        def jac(x_flat):
            x = x_flat.reshape(N, 3)
            _, g = _energy_and_grad(x)
            return g.flatten()
        
        # Use L-BFGS-B with equality constraint for sphere
        # Actually, let's use a different approach: parameterize on sphere
        
        # For now, use SLSQP with constraints
        constraints = []
        for i in range(N):
            constraints.append({
                'type': 'eq',
                'fun': lambda x_flat, i=i: np.dot(x_flat[3*i:3*i+3], x_flat[3*i:3*i+3]) - 1.0,
                'jac': lambda x_flat, i=i: _sphere_jac(x_flat, i)
            })
        
        res = minimize(
            fun, x0.flatten(), jac=jac,
            method='SLSQP', constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < _energy_fast(x0):
            x_best = _project_to_sphere(res.x.reshape(N, 3))
        else:
            x_best = x0
            
    except Exception:
        x_best = x0
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining < 5.0:
        return x_best, _energy_fast(x_best)
    
    # Riemannian gradient descent polish
    x = x_best.copy()
    e_prev = _energy_fast(x)
    
    # Adaptive step size
    lr = 0.01
    lr_min = 1e-8
    lr_max = 0.1
    patience = 50
    no_improve = 0
    
    while time.time() - start_time < time_limit - 3.0:
        # Compute gradient
        _, grad = _energy_and_grad(x)
        
        # Project gradient to tangent space
        # Tangent space: v such that v . x_i = 0 for each i
        grad_tangent = grad - (grad * x).sum(axis=1, keepdims=True) * x
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
        
        # Try step
        x_new = x + lr * grad_tangent
        x_new = _project_to_sphere(x_new)
        
        e_new = _energy_fast(x_new)
        
        if e_new < e_prev:
            x = x_new
            e_prev = e_new
            no_improve = 0
            lr = min(lr * 1.1, lr_max)
        else:
            no_improve += 1
            lr = max(lr * 0.5, lr_min)
            if no_improve >= patience:
                break
    
    return x, e_prev


def _sphere_jac(x_flat, i):
    """Jacobian for sphere constraint of point i."""
    j = np.zeros_like(x_flat)
    j[3*i:3*i+3] = 2.0 * x_flat[3*i:3*i+3]
    return j


def _perturb(x, rng, scale=0.1):
    """Add random perturbation and project back to sphere."""
    noise = rng.standard_normal(x.shape) * scale
    return _project_to_sphere(x + noise)


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    time_budget = MAX_TIME - (time.time() - start_time)
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    # Generate diverse starting configurations
    starts = []
    
    # Golden spiral with various offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        starts.append(_golden_spiral(offset))
    
    # Random starts
    for _ in range(5):
        starts.append(_random_points(rng))
    
    # Fibonacci lattice variations
    for _ in range(3):
        starts.append(_random_points(rng))
    
    # Try each start with time allocation
    n_starts = len(starts)
    time_per_start = time_budget / (n_starts + 2)  # +2 for final polish
    
    for idx, x0 in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        alloc = min(time_per_start, remaining - 2.0)
        if alloc < 1.0:
            break
        
        x_opt, e_opt = _optimize_from_start(x0, elapsed + alloc, rng, label=f"start{idx}")
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # If we found a good solution, try perturbation-based escape
        if idx >= 2 and time.time() - start_time < start_time + time_budget - 5.0:
            x_pert = _perturb(x_opt, rng, scale=0.05)
            x_opt2, e_opt2 = _optimize_from_start(x_pert, time.time() + 10.0, rng, label=f"perturb{idx}")
            if e_opt2 < best_e:
                best_e = e_opt2
                best_x = x_opt2.copy()
    
    # Final high-precision polish on the best solution
    if best_x is not None and time.time() - start_time < start_time + time_budget - 5.0:
        x_final, e_final = _optimize_from_start(best_x, time.time() + (time_budget - (time.time() - start_time) - 2.0), rng, label="final")
        if e_final < best_e:
            best_e = e_final
            best_x = x_final.copy()
    
    # Ensure validity
    if best_x is None:
        best_x = _golden_spiral(0.0)
    
    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Check for points too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                # Nudge apart
                best_x[j] += 1e-8 * rng.standard_normal(3)
                best_x = _project_to_sphere(best_x)
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _golden_spiral(0.0)
    
    return best_x

# EVOLVE-BLOCK-END
