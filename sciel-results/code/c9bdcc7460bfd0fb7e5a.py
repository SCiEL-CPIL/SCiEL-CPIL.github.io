import time
import math
import numpy as np
from scipy.spatial.distance import cdist

N = 100
_DEADLINE = time.time() + 115.0  # Safety margin

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the sphere.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    """
    # Compute pairwise distances
    # d[i, j] = ||x_i - x_j||
    d = cdist(pts, pts)
    
    # Avoid log(0) and division by zero
    # We know d[i,i] = 0. We mask the diagonal.
    mask = ~np.eye(N, dtype=bool)
    d_safe = np.where(mask, d, 1.0)
    
    # Energy: sum of -log(d) for i<j
    # We can sum all i!=j and divide by 2
    log_d = np.log(d_safe)
    energy = -0.5 * np.sum(log_d)
    
    # Gradient
    # grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # Let's compute the matrix of differences
    # diff[i, j] = x_i - x_j
    # This is a bit memory heavy for large N, but N=100 is small.
    # diff shape (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # We need to divide by d_ij^2
    # d_safe shape (N, N)
    # We want inv_d2[i, j] = 1 / d_ij^2 if i!=j else 0
    inv_d2 = np.zeros((N, N))
    inv_d2[mask] = 1.0 / (d_safe[mask] ** 2)
    
    # grad[i] = sum_j inv_d2[i, j] * diff[i, j]
    # This is a matrix multiplication: inv_d2 @ diff
    # But diff is (N, N, 3). We can reshape or use einsum.
    # grad[i, k] = sum_j inv_d2[i, j] * diff[i, j, k]
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)
    
    # Project gradient to tangent space
    # g_tan = g - (g . x) x
    g_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - g_dot_x * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibsphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _optimize_riemannian(pts0, max_iter=1000, tol=1e-10):
    """
    Riemannian Gradient Descent with backtracking line search.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_e, _ = _energy_and_grad(pts)
    
    step = 0.01
    patience = 0
    max_patience = 100
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 2:
            break
            
        e, g_tan = _energy_and_grad(pts)
        if e < best_e - 1e-12:
            best_e = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
                
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
            
        # Backtracking line search
        alpha = step
        improved = False
        for _ in range(20):
            # Step in tangent direction
            new_pts = pts + alpha * g_tan
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _energy_and_grad(new_pts)
            
            if new_e < e:
                pts = new_pts
                improved = True
                break
            alpha *= 0.5
            
        if not improved:
            # If even very small step doesn't improve, we are stuck or at minimum
            break
            
        # Adjust step size
        if improved:
            step = min(step * 1.1, 0.1)
        else:
            step *= 0.5
            if step < 1e-10:
                break
                
    return best_pts, best_e

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    best_pts = None
    best_e = float('inf')
    
    # Initializations
    inits = []
    inits.append(_fibsphere(N))
    inits.append(_random_sphere(N, seed=1))
    inits.append(_random_sphere(N, seed=2))
    
    # Add perturbed Fibonacci
    fib = _fibsphere(N)
    for i in range(3):
        rng = np.random.RandomState(100+i)
        noise = rng.randn(N, 3) * 0.1
        perturbed = _project_to_sphere(fib + noise)
        inits.append(perturbed)
        
    for init in inits:
        if time.time() > _DEADLINE - 10:
            break
        pts, e = _optimize_riemannian(init, max_iter=500, tol=1e-8)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            
    # Final polish on the best candidate
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts, e = _optimize_riemannian(best_pts, max_iter=1000, tol=1e-12)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    d = cdist(best_pts, best_pts)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(d[iu])
    if min_dist < 1e-9:
        # Fallback
        best_pts = _fibsphere(N)
        
    return best_pts
