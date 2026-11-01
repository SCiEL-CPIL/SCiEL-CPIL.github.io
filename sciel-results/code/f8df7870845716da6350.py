import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_grid(n_grid=20000):
    """Generate a dense spherical grid for the surrogate objective."""
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_radius(pts):
    """Compute the exact covering radius using the convex hull method."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
        
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
    except Exception:
        return np.pi

def _soft_hull_objective_and_grad(pts, grid, alpha=50.0):
    """
    Compute the soft-hull objective (log-sum-exp approximation of max distance)
    and its gradient with respect to points on the sphere.
    
    Objective: J = (1/alpha) * log(sum(exp(-alpha * d_i))) where d_i is min distance to grid point i.
    Actually, we want to MINIMIZE the max distance. 
    Let d_i be the distance from grid point y_i to nearest point x_k.
    We want min max_i d_i.
    A smooth approximation is: f = (1/alpha) log sum_i exp(alpha * d_i).
    Gradient of f w.r.t x_k:
    df/dx_k = sum_i (alpha * exp(alpha * d_i) / sum_j exp(alpha * d_j)) * (d/dx_k d_i)
    d_i = arccos(max_j dot(y_i, x_j))
    Let j* be argmax_j dot(y_i, x_j).
    d/dx_k d_i = 0 if k != j*.
    If k == j*, d/dx_k arccos(dot(y_i, x_k)) = - dot(y_i, x_k) / sqrt(1 - dot(y_i, x_k)^2) * (y_i - dot(y_i, x_k)x_k) / ||y_i - dot(y_i, x_k)x_k||?
    No, simpler: d = arccos(c), c = dot(y, x). dd/dx = -1/sqrt(1-c^2) * (y - c*x).
    Note: sqrt(1-c^2) = sin(d).
    So dd/dx = -(y - c*x) / sin(d).
    
    We use this to update x_k on the sphere.
    """
    # Compute distances from each grid point to each candidate point
    # d[i, k] = arccos(dot(grid[i], pts[k]))
    dots = grid @ pts.T  # (G, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (G, N)
    
    # min distance for each grid point
    min_dists = np.min(dists, axis=1)  # (G,)
    
    # For gradient, we need the index of the nearest point for each grid point
    # And the value of the min distance
    nearest_idx = np.argmin(dists, axis=1)
    
    # Soft max approximation
    # f = (1/alpha) * log(sum(exp(alpha * min_dists)))
    # To avoid overflow, subtract max
    max_d = np.max(min_dists)
    exps = np.exp(alpha * (min_dists - max_d))
    sum_exps = np.sum(exps)
    if sum_exps < _EPS:
        sum_exps = _EPS
    
    f = (1.0 / alpha) * (np.log(sum_exps) + alpha * max_d)
    
    # Gradient
    # grad_f_k = sum_i (alpha * exps[i] / sum_exps) * (d/dx_k min_d_i)
    # d/dx_k min_d_i is non-zero only if k == nearest_idx[i]
    # d/dx_k min_d_i = -(y_i - dot(y_i, x_k) x_k) / sin(min_d_i)
    
    weights = alpha * exps / sum_exps  # (G,)
    
    grad = np.zeros_like(pts)
    
    # For each grid point i, we add to grad[nearest_idx[i]] the term:
    # weights[i] * (-(y_i - dot_i x_k) / sin(d_i))
    # where dot_i = dots[i, nearest_idx[i]] and d_i = min_dists[i]
    
    y_i = grid  # (G, 3)
    x_k = pts[nearest_idx]  # (G, 3)
    dot_i = dots[np.arange(len(grid)), nearest_idx]  # (G,)
    d_i = min_dists  # (G,)
    
    sin_d = np.sin(d_i)
    # Avoid division by zero
    sin_d = np.where(sin_d < _EPS, _EPS, sin_d)
    
    # Term: (y_i - dot_i x_k) / sin_d
    # shape (G, 3)
    term = (y_i - dot_i[:, np.newaxis] * x_k) / sin_d[:, np.newaxis]
    
    # grad_k = sum_i weights[i] * (-term_i)
    # We need to scatter-add these to grad
    # Use bincount or simple loop if G is small, but G=20000, N=130.
    # We can use np.add.at
    np.add.at(grad, nearest_idx, -weights[:, np.newaxis] * term)
    
    return f, grad

def _riemannian_grad_descent(pts, grid, alpha=50.0, lr=0.01, n_iter=1000, deadline=None):
    """
    Optimize points on the sphere using Riemannian gradient descent.
    Update: x_new = normalize(x - lr * grad)
    """
    best_pts = pts.copy()
    best_f = _soft_hull_objective_and_grad(pts, grid, alpha)[0]
    
    for it in range(n_iter):
        if deadline and time.time() > deadline:
            break
            
        f, grad = _soft_hull_objective_and_grad(pts, grid, alpha)
        
        # Riemannian gradient step
        # Project gradient to tangent space: g_tan = grad - (grad . x) x
        # Then update: x_new = exp_x(-lr * g_tan) ≈ normalize(x - lr * g_tan)
        
        dot_grad_x = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tan = grad - dot_grad_x * pts
        
        new_pts = pts - lr * grad_tan
        new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
        
        # Check if improved
        new_f, _ = _soft_hull_objective_and_grad(new_pts, grid, alpha)
        
        if new_f < best_f:
            best_f = new_f
            best_pts = new_pts.copy()
            pts = new_pts
            # Adaptive learning rate: increase if progress is good
            if it % 50 == 0 and it > 0:
                lr *= 1.1
                lr = min(lr, 0.1)
        else:
            # Decrease learning rate
            lr *= 0.8
            if lr < 1e-6:
                break
    
    return best_pts, best_f

def _polish_with_hull(pts, n_iter=50, deadline=None):
    """
    Use the exact hull radius to polish the result.
    Simple pole-chasing on the worst facets.
    """
    best_pts = pts.copy()
    best_r = _hull_radius(pts)
    
    for it in range(n_iter):
        if deadline and time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii = np.arccos(cos_theta)
            
            curr_r = np.max(radii)
            if curr_r < best_r - 1e-12:
                best_r = curr_r
                best_pts = pts.copy()
                
            worst_idx = np.argmax(radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            step_size = 0.05
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    continue
                cand[v] = pts[v] + step_size * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            cand_r = _hull_radius(cand)
            if cand_r < curr_r - 1e-14:
                pts = cand
            else:
                break
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    
    grid = _get_grid(20000)
    
    best_pts = _initial_fib()
    best_r = _hull_radius(best_pts)
    
    # Phase 1: Riemannian Gradient Descent with multiple restarts
    n_restarts = 10
    time_per_restart = (deadline - 30.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Create perturbed starting point
        if restart == 0:
            pts = best_pts.copy()
        else:
            rng = np.random.default_rng(100 + restart * 7)
            pts = best_pts + rng.standard_normal((N, 3)) * 0.2
            pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
            
        restart_deadline = min(deadline - 30.0, time.time() + time_per_restart)
        
        # Run Riemannian GD
        pts, f = _riemannian_grad_descent(pts, grid, alpha=50.0, lr=0.05, n_iter=500, deadline=restart_deadline)
        
        # Polish with exact hull
        pts, r = _polish_with_hull(pts, n_iter=50, deadline=deadline - 25.0)
        
        if r < best_r - 1e-12:
            best_r = r
            best_pts = pts
    
    # Final polish
    if time.time() < deadline - 5.0:
        best_pts, best_r = _polish_with_hull(best_pts, n_iter=100, deadline=deadline - 2.0)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
