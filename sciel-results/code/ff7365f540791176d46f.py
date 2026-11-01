"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on unit sphere.
    
    Energy: sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / dist[i,j]^2
    # diff[i,j] = x_i - x_j, so we need sum over j of diff[i,j] / dist[i,j]^2
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _riemannian_descent(pts, max_iter=5000, tol=1e-12, initial_lr=0.1):
    """Riemannian gradient descent with backtracking line search.
    
    At each step:
    1. Compute gradient in ambient space
    2. Project to tangent space: grad_t = grad - <grad, x> * x
    3. Retraction: x_new = normalize(x + step * grad_t)
    4. Backtracking line search for step size
    """
    pts = _project_to_sphere(pts.copy())
    n = pts.shape[0]
    
    f, g = _energy_and_grad(pts)
    
    # Initial step size
    lr = initial_lr
    
    for it in range(max_iter):
        # Project gradient to tangent space
        inner = np.sum(pts * g, axis=1, keepdims=True)
        grad_t = g - inner * pts
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = lr
        f_new = None
        for _ in range(30):
            # Retraction: move along tangent direction and renormalize
            new_pts = _project_to_sphere(pts + alpha * grad_t)
            f_new, _ = _energy_and_grad(new_pts)
            
            if f_new < f - 1e-4 * alpha * np.sum(pts * grad_t):
                break
            alpha *= 0.5
        else:
            # If line search fails, accept anyway
            pass
        
        pts = new_pts
        f = f_new
        g, _ = _energy_and_grad(pts)
        
        # Adapt learning rate
        if f_new < f - 1e-10:
            lr = min(lr * 1.5, 1.0)
        else:
            lr = max(lr * 0.8, 1e-6)
    
    return pts, f


def _lbfgs_optimize(pts, max_iter=10000, ftol=1e-14, gtol=1e-12):
    """L-BFGS optimization with sphere constraint via projection.
    
    We optimize in ambient R^(3N) and project back to sphere after each step.
    """
    pts = _project_to_sphere(pts.copy())
    
    def objective_and_grad(flat):
        p = flat.reshape(N, 3)
        p = _project_to_sphere(p)
        f, g = _energy_and_grad(p)
        # Project gradient to tangent space for correct gradient on manifold
        inner = np.sum(p * g, axis=1, keepdims=True)
        g_t = g - inner * p
        return f, g_t.flatten()
    
    x0 = pts.flatten()
    res = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    
    best_pts = _project_to_sphere(res.x.reshape(N, 3))
    return best_pts, res.fun


def _fibonacci_sphere(n, seed_offset=0):
    """Generate n points on sphere using Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _verify(pts):
    """Check if points are valid."""
    if pts.shape != (N, 3):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-15):
        return False
    # Check minimum distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 1e-9:
        return False
    return True


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            pts = _fibonacci_sphere(N)
        elif restart == 1:
            pts = _fibonacci_sphere(N, seed_offset=0.3)
        elif restart == 2:
            pts = _fibonacci_sphere(N, seed_offset=0.7)
        else:
            pts = _random_sphere(N, rng)
        
        pts = _project_to_sphere(pts)
        
        # Phase 1: Riemannian gradient descent for coarse optimization
        if time.time() < deadline - 20:
            pts, f1 = _riemannian_descent(pts, max_iter=3000, tol=1e-10, initial_lr=0.05)
        
        # Phase 2: L-BFGS for fine optimization
        if time.time() < deadline - 5:
            pts, f2 = _lbfgs_optimize(pts, max_iter=5000, ftol=1e-14, gtol=1e-12)
        
        if _verify(pts):
            f, _ = _energy_and_grad(pts)
            if f < best_energy:
                best_energy = f
                best_pts = pts.copy()
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 2 and _verify(best_pts):
        best_pts, f_final = _lbfgs_optimize(best_pts, max_iter=10000, ftol=1e-15, gtol=1e-13)
        if _verify(best_pts):
            f_check, _ = _energy_and_grad(best_pts)
            if f_check < best_energy:
                best_energy = f_check
    
    # Ensure final output is on the unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Final verification
    if not _verify(best_pts):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
