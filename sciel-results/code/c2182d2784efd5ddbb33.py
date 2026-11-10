# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3
    start_time = time.time()
    timeout = 580.0

    def compute_score(points):
        """Compute the actual metric: (d_min / d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or d_min == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def fib_sphere(n):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def gradient_descent_sphere(points, iterations=5000, lr=0.01, momentum=0.9):
        """
        Minimizes sum(1/d_ij^2) on the sphere using gradient descent with projection.
        """
        pts = points.copy()
        v = np.zeros_like(pts)
        
        for _ in range(iterations):
            if time.time() - start_time > timeout - 10:
                break
                
            # Compute pairwise distances squared
            dists_sq = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2)
            
            # Avoid division by zero
            dists_sq = np.maximum(dists_sq, 1e-12)
            
            # Gradient of sum_{j != i} 1/d_ij^2 w.r.t p_i
            # term_ij = -2 * (p_i - p_j) / d_ij^4
            inv_d4 = 1.0 / (dists_sq ** 2)
            
            # Set diagonal to 0 to avoid self-interaction
            np.fill_diagonal(inv_d4, 0.0)
            
            # grad_i = -2 * sum_j (p_i - p_j) * inv_d4[i, j]
            sum_inv = np.sum(inv_d4, axis=1)
            weighted_pos = np.sum(pts * inv_d4[:, :, np.newaxis], axis=1)
            
            grads = -2.0 * (pts * sum_inv[:, np.newaxis] - weighted_pos)
            
            # Update velocity with momentum
            v = momentum * v - lr * grads
            
            # Update positions
            pts = pts + v
            
            # Project back to sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pts = pts / norms
            
        return pts

    def refine_with_nm(points):
        """Refine using Nelder-Mead on the actual objective."""
        def objective(x):
            if time.time() - start_time > timeout - 5:
                raise TimeoutError()
            pts = x.reshape(n, d)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pts = pts / norms
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max == 0 or d_min == 0:
                return 1e6
            return - (d_min / d_max) ** 2

        try:
            res = minimize(objective, points.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'maxfev': 1000})
            res_pts = res.x.reshape(n, d)
            norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            res_pts = res_pts / norms
            return res_pts
        except Exception:
            return points

    best_points = None
    best_score = -1e9
    
    # Initializations
    inits = []
    # 1. Fibonacci Sphere
    inits.append(fib_sphere(n))
    
    # 2. Random
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts)
        
    # 3. Perturbed Fibonacci
    for _ in range(5):
        pts = fib_sphere(n) + np.random.randn(n, d) * 0.1
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts)

    # Run gradient descent for each init
    for init in inits:
        if time.time() - start_time > timeout - 60:
            break
            
        # Try a few different learning rates
        for lr in [0.005, 0.01, 0.02]:
            if time.time() - start_time > timeout - 30:
                break
                
            optimized = gradient_descent_sphere(init, iterations=2000, lr=lr, momentum=0.9)
            
            # Refine with a finer step
            optimized = gradient_descent_sphere(optimized, iterations=1000, lr=lr*0.1, momentum=0.9)
            
            # Final polish with Nelder-Mead on the true objective
            if time.time() - start_time < timeout - 15:
                optimized = refine_with_nm(optimized)
            
            score = compute_score(optimized)
            
            if score > best_score:
                best_score = score
                best_points = optimized

    # Fallback
    if best_points is None:
        best_points = fib_sphere(n)
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
