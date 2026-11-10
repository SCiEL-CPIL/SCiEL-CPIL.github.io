"""Minimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # Safety margin before 150s kill

def _log_energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) and gradient.
    
    pts: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    """
    # Compute pairwise distances
    # d_ij = |x_i - x_j|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu]
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|)
    # = d/dx_i [-sum_{j!=i} log(|x_i - x_j|)]
    # = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # 
    # grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: for log(1/d) = -log(d), d(-log(d))/dx = -(1/d) * d(d)/dx
    # d(d)/dx_i = (x_i - x_j)/d, so d(-log(d))/dx_i = -(1/d)*(x_i-x_j)/d = -(x_i-x_j)/d^2
    # Wait, let me recompute:
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j!=i} d(log(d_ij))/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij
    #          = -sum_{j!=i} (x_i - x_j) / d_ij^2
    # 
    # But we want to MINIMIZE E, so gradient descent uses -grad.
    # Actually for minimization, we compute grad of E and do x -= lr * grad.
    
    # Compute gradient efficiently
    # grad[i] = -sum_{j != i} (x_i - x_j) / d_ij^2
    # = sum_{j != i} (x_j - x_i) / d_ij^2
    
    # Using matrix operations:
    # inv_d2 = 1/d^2 for off-diagonal
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) * inv_d2[i,j]
    # = sum_j x_j * inv_d2[i,j] - x_i * sum_j inv_d2[i,j]
    # = (inv_d2 @ pts).T ... let me think carefully
    # 
    # grad[i,k] = sum_j (pts[j,k] - pts[i,k]) * inv_d2[i,j]
    #            = sum_j pts[j,k] * inv_d2[i,j] - pts[i,k] * sum_j inv_d2[i,j]
    # 
    # Let A = inv_d2 (N x N), row sums s = A.sum(axis=1)
    # grad = A.T @ pts - pts * s[:, np.newaxis]
    # Wait: (A.T @ pts)[i,k] = sum_j A.T[i,j] * pts[j,k] = sum_j A[j,i] * pts[j,k]
    # That's not quite right. Let me use:
    # grad[i] = sum_j (pts[j] - pts[i]) * A[i,j]
    # = sum_j pts[j] * A[i,j] - pts[i] * sum_j A[i,j]
    # = (A @ pts).T ... no
    # 
    # (A @ pts): (A @ pts)[j,k] = sum_i A[j,i] * pts[i,k] -- not what we want
    # 
    # We want: for each i, sum_j A[i,j] * pts[j]
    # That's (A @ pts) where A is (N,N) and pts is (N,3):
    # (A @ pts)[i,k] = sum_j A[i,j] * pts[j,k] ✓
    
    row_sums = inv_d2.sum(axis=1)  # (N,)
    grad = A_pts = inv_d2 @ pts  # (N, 3)
    grad = grad - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts0, time_budget):
    """Run L-BFGS optimization from initial points. Returns best points found."""
    best_pts = pts0.copy()
    best_energy = _log_energy_and_grad(pts0)[0]
    
    # Use L-BFGS-B with projection via penalty or just L-BFGS on flattened
    # We'll use a custom approach: L-BFGS on the manifold via projection
    # Simpler: use scipy minimize with L-BFGS-B, and project after each step
    
    x0 = pts0.flatten()
    
    def fun_and_grad(x):
        pts = x.reshape(N, 3)
        # Project to sphere for evaluation
        pts_p = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts_p)
        # For the gradient in the ambient space, we need to account for projection
        # But L-BFGS will handle it approximately. Alternatively, use the gradient
        # on the manifold projected to ambient space.
        # The Riemannian gradient projected to ambient space is just the Euclidean
        # gradient minus its normal component. Since we're on the sphere, the normal
        # is x/|x| = x (if |x|=1).
        # grad_ambient = grad - (grad . x) * x
        grad_proj = g - (g * pts_p).sum(axis=1, keepdims=True) * pts_p
        return e, grad_proj.flatten()
    
    try:
        res = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 50,
                'maxfun': 5000,
            }
        )
        best_pts = _project_to_sphere(res.x.reshape(N, 3))
        best_energy = _log_energy_and_grad(best_pts)[0]
    except Exception:
        pass
    
    # Additional refinement with more iterations if time allows
    if time.time() < time_budget:
        try:
            x1 = best_pts.flatten()
            res2 = minimize(
                fun_and_grad,
                x1,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 50,
                    'maxfun': 10000,
                }
            )
            pts2 = _project_to_sphere(res2.x.reshape(N, 3))
            e2 = _log_energy_and_grad(pts2)[0]
            if e2 < best_energy:
                best_pts = pts2
                best_energy = e2
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Fibonacci spiral
    pts_fib = _fibonacci_spiral()
    best_pts, best_energy = _optimize_from_start(pts_fib, deadline - 2.0)
    
    # Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    for seed in [1, 7, 13, 21, 34]:
        if time.time() >= deadline - 30.0:
            break
        pts_rand = _random_points(seed)
        try:
            pts_opt, e_opt = _optimize_from_start(pts_rand, deadline - 2.0)
            if e_opt < best_energy:
                best_pts = pts_opt
                best_energy = e_opt
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            x0 = best_pts.flatten()
            
            def fun_and_grad_p(x):
                pts = _project_to_sphere(x.reshape(N, 3))
                e, g = _log_energy_and_grad(pts)
                grad_proj = g - (g * pts).sum(axis=1, keepdims=True) * pts
                return e, grad_proj.flatten()
            
            res = minimize(
                fun_and_grad_p,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-16,
                    'gtol': 1e-15,
                    'maxls': 50,
                    'maxfun': 20000,
                }
            )
            pts_final = _project_to_sphere(res.x.reshape(N, 3))
            e_final = _log_energy_and_grad(pts_final)[0]
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure all points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += 1e-8 * rng.randn(3)
                best_pts[j] = _project_to_sphere(best_pts[j])
    
    return best_pts
