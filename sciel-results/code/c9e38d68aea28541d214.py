"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for point k, grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # But we need to be careful: energy = -sum_{i<j} log(dist_ij)
    # d/dx_k [-log(dist_kj)] = -(x_k - x_j)/dist_kj^2 for j != k
    # Actually: d/dx_k log(dist_kj) = (x_k - x_j)/dist_kj^2
    # So d/dx_k [-log(dist_kj)] = -(x_k - x_j)/dist_kj^2
    # Total gradient for point k: sum_{j != k} -(x_k - x_j)/dist_kj^2
    # But wait, each pair (i,j) with i<j contributes to both x_i and x_j.
    # For point k, it appears in pairs (k,j) for j>k and pairs (i,k) for i<k.
    # In both cases, the contribution to grad_k is -(x_k - x_j)/dist_kj^2
    
    # Efficient computation:
    # grad_k = -sum_{j != k} (x_k - x_j) / dist_kj^2
    # = -n*x_k / 1 + sum_{j != k} x_j / dist_kj^2 ... let me redo
    
    # Actually: sum_{j != k} (x_k - x_j)/dist_kj^2 = x_k * sum_{j!=k} 1/dist_kj^2 - sum_{j!=k} x_j/dist_kj^2
    # Let's compute directly using matrix operations
    
    # dist is n x n, diagonal set to 1
    # For gradient, we need: grad[k] = -sum_{j != k} (pts[k] - pts[j]) / dist[k,j]^2
    
    # Create weight matrix: W[k,j] = 1/dist[k,j]^2 for k != j, 0 otherwise
    W = np.zeros_like(dist)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / (dist ** 2)
    np.fill_diagonal(W, 0.0)
    
    # grad[k] = -sum_j W[k,j] * (pts[k] - pts[j])
    # = -sum_j W[k,j]*pts[k] + sum_j W[k,j]*pts[j]
    # = -pts[k] * sum_j W[k,j] + sum_j W[k,j]*pts[j]
    row_sums = W.sum(axis=1)  # shape (n,)
    grad = -pts * row_sums[:, np.newaxis] + W @ pts
    
    return energy, grad


def _exponential_map(p, v, step=1.0):
    """Exponential map on the unit sphere: move from p in direction v by step.
    v should be a tangent vector (orthogonal to p).
    """
    p = p / np.linalg.norm(p)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-15:
        return p
    v_hat = v / v_norm
    # Make v tangent
    v_tan = v_hat - np.dot(v_hat, p) * p
    v_tan_norm = np.linalg.norm(v_tan)
    if v_tan_norm < 1e-15:
        return p
    v_tan_hat = v_tan / v_tan_norm
    angle = step * v_tan_norm / v_tan_norm  # = step
    # Actually, the step size in the tangent direction
    # exp_p(v) = cos(|v|) * p + sin(|v|) * v/|v|
    # where v is the tangent vector
    theta = step
    return np.cos(theta) * p + np.sin(theta) * v_tan_hat


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points():
    """Generate random points on the sphere."""
    pts = np.random.randn(N, 3)
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, maxiter=500, gtol=1e-10):
    """Optimize using L-BFGS-B with sphere constraint via projection."""
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # Gradient with respect to x (before projection) - approximate by using pts gradient
        # This is a simplification; for better results, we'd need the full chain rule
        return e, g.reshape(-1)
    
    result = minimize(objective_and_grad, pts.reshape(-1), method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
    return _project_to_sphere(result.x.reshape(N, 3)), result.fun


def _cg_on_sphere(pts, max_iter=100, tol=1e-10):
    """Conjugate gradient on the sphere using exponential maps."""
    pts = _project_to_sphere(pts)
    energy, grad = _energy_and_grad(pts)
    
    # Initial direction: negative gradient projected to tangent space
    d = -grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    # Project d to tangent space: d = d - (d.p)*p
    d = d - np.sum(d * pts, axis=1, keepdims=True) * pts
    
    best_pts = pts.copy()
    best_energy = energy
    
    for it in range(max_iter):
        # Line search: find good step size
        alpha = 1.0
        for _ in range(50):
            new_pts = np.array([_exponential_map(pts[i], d[i], alpha) for i in range(N)])
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                break
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        if alpha < 1e-10:
            break
            
        old_grad = grad
        pts = new_pts
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_pts = pts.copy()
            best_energy = energy
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Update direction (Polak-Ribiere)
        beta = np.sum((grad - old_grad) * grad, axis=1) / (np.sum(old_grad * old_grad, axis=1) + 1e-15)
        beta = np.maximum(beta, 0)
        d = -grad + beta[:, np.newaxis] * d
        # Project to tangent space
        d = d - np.sum(d * pts, axis=1, keepdims=True) * pts
    
    return _project_to_sphere(best_pts), best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Multiple initializations and optimization
    best_pts = _fibonacci_spiral()
    best_energy, _ = _energy_and_grad(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Try several initializations
    initializations = [_fibonacci_spiral()]
    for _ in range(5):
        initializations.append(_random_points())
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        # Phase 1: L-BFGS-B
        pts, e = _lbfgs_optimize(init, maxiter=300, gtol=1e-10)
        if e < best_energy:
            best_energy = e
            best_pts = pts
        
        if time.time() > deadline:
            break
        
        # Phase 2: Conjugate gradient on sphere
        pts, e = _cg_on_sphere(pts, max_iter=200, tol=1e-12)
        if e < best_energy:
            best_energy = e
            best_pts = pts
        
        if time.time() > deadline:
            break
        
        # Phase 3: Another L-BFGS-B pass
        pts, e = _lbfgs_optimize(pts, maxiter=300, gtol=1e-12)
        if e < best_energy:
            best_energy = e
            best_pts = pts
        
        if time.time() > deadline:
            break
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral()
    
    return best_pts
# EVOLVE-BLOCK-END
