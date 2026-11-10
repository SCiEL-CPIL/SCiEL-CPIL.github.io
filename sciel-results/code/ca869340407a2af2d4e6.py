"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
SEED = 42


def _energy_and_grad(x):
    """Compute logarithmic energy and gradient for points on sphere.
    x: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the Euclidean gradient (not projected).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    energy = -np.sum(log_d)
    
    # Gradient: d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2 for each pair
    # grad_i = sum_{j != i} (x_i - x_j) / d_ij^2  ... wait let me re-derive
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} d/dx_i log(d_ij) - sum_{j<i} d/dx_i log(d_ij)
    # d/dx_i log(d_ij) = (x_i - x_j) / d_ij^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j) / d_ij^2 = sum_{j != i} (x_j - x_i) / d_ij^2
    
    grad = np.zeros_like(x)
    # For each pair (i,j), contribution to grad[i] is (x_j - x_i)/d^2, to grad[j] is (x_i - x_j)/d^2
    # Vectorized: for all pairs
    # diff[i,j] = x_i - x_j
    # grad contribution from pair (i,j) to grad[i]: (x_j - x_i)/d^2 = -diff[i,j]/d^2
    # grad contribution from pair (i,j) to grad[j]: (x_i - x_j)/d^2 = diff[i,j]/d^2
    
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) / d_ij^2 = sum_j -diff[i,j] * inv_d2[i,j]
    # = -sum_j diff[i,j] * inv_d2[i,j]
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project(x):
    """Project points to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(rng):
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _ring_based(rng, n_rings=10):
    """Create points in rings with some randomness."""
    pts = []
    for ring in range(n_rings):
        z = 1.0 - 2.0 * (ring + 0.5) / n_rings
        r = np.sqrt(1.0 - z * z)
        n_in_ring = N // n_rings + (1 if ring < N % n_rings else 0)
        for k in range(n_in_ring):
            phi = 2.0 * np.pi * k / n_in_ring + rng.uniform(-0.1, 0.1)
            pts.append([r * np.cos(phi), r * np.sin(phi), z])
    return np.array(pts[:N])


def _optimize_from(x0, time_budget, rng):
    """Optimize from initial point x0, returning best found and its energy."""
    x0 = _project(x0.copy())
    e0, _ = _energy_and_grad(x0)
    best_x = x0.copy()
    best_e = e0
    
    # Use L-BFGS-B on the sphere via parameterization or direct with projection
    # Direct approach: minimize on R^3 with projection after each step
    # Better: use scipy minimize with a custom approach
    
    # We'll use L-BFGS-B on the unconstrained problem but project after each iteration
    # Actually, let's use the fact that we can do gradient descent with projection
    
    # First try scipy minimize with bounds (each coordinate in [-1,1])
    # But that's not quite right for sphere constraint.
    
    # Better: use L-BFGS-B on the sphere by working in R^3 and projecting
    # scipy's minimize doesn't handle equality constraints well with L-BFGS-B
    
    # Let's do manual gradient descent with adaptive step size + L-BFGS polish
    
    # Phase 1: Gradient descent with backtracking line search
    lr = 0.01
    for step in range(500):
        if time.time() > time_budget:
            break
        x = _project(x0)
        e, g = _energy_and_grad(x)
        
        if e < best_e:
            best_e = e
            best_x = x.copy()
        
        # Gradient descent step
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
        
        # Adaptive learning rate based on gradient norm
        lr_eff = lr / (1.0 + 0.1 * g_norm)
        
        # Backtracking line search
        alpha = 1.0
        for _ in range(30):
            x_new = _project(x - alpha * lr_eff * g)
            e_new, _ = _energy_and_grad(x_new)
            if e_new < e:
                x0 = x_new
                break
            alpha *= 0.5
        else:
            # If no improvement, break
            break
    
    # Phase 2: L-BFGS-B polish using spherical coordinates or direct with tight tol
    # Use scipy minimize with the projected gradient idea
    # We'll parameterize using tangent space: at each point, work in the tangent plane
    # Simpler: just run L-BFGS-B on R^3 with projection as a wrapper
    
    def objective_and_grad_flat(flat_x):
        x = _project(flat_x.reshape(N, 3))
        e, g = _energy_and_grad(x)
        # The gradient for the flat problem needs to account for projection
        # For now, just use the Euclidean gradient as an approximation
        return e, g.ravel()
    
    try:
        res = minimize(
            lambda fx: _energy_and_grad(_project(fx.reshape(N, 3)))[0],
            x0.ravel(),
            jac=lambda fx: _energy_and_grad(_project(fx.reshape(N, 3)))[1].ravel(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        if res.success or res.fun < best_e:
            x_polish = _project(res.x.reshape(N, 3))
            e_polish, _ = _energy_and_grad(x_polish)
            if e_polish < best_e:
                best_e = e_polish
                best_x = x_polish
    except Exception:
        pass
    
    return best_x, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    rng = np.random.default_rng(SEED)
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral())
    for _ in range(3):
        inits.append(_random_points(rng))
    for _ in range(2):
        inits.append(_ring_based(rng))
    
    # Also add perturbed fibonacci
    for _ in range(2):
        fib = _fibonacci_spiral()
        perturbed = fib + rng.standard_normal((N, 3)) * 0.1
        inits.append(_project(perturbed))
    
    best_x = None
    best_e = float('inf')
    
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        
        x_opt, e_opt = _optimize_from(init, deadline - 10.0, rng)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt
    
    # Final polish on the best result with tighter tolerance
    if best_x is not None and time.time() < deadline - 5.0:
        try:
            def obj_flat(fx):
                return _energy_and_grad(_project(fx.reshape(N, 3)))[0]
            
            def jac_flat(fx):
                return _energy_and_grad(_project(fx.reshape(N, 3)))[1].ravel()
            
            res = minimize(
                obj_flat,
                best_x.ravel(),
                jac=jac_flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-15, 'ftol': 1e-18}
            )
            if res.fun < best_e:
                best_x = _project(res.x.reshape(N, 3))
                best_e, _ = _energy_and_grad(best_x)
        except Exception:
            pass
    
    # Validate
    best_x = _project(best_x)
    # Check for very close points
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        # Fallback to fibonacci
        best_x = _fibonacci_spiral()
    
    return best_x
