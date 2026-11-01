import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130
DEADLINE = 145.0  # safety margin before 150s kill

def _covering_radius_from_hull(points):
    """
    Compute the exact covering radius from the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        normal = np.cross(b - a, c - a)
        norm_val = np.linalg.norm(normal)
        
        if norm_val < 1e-12:
            continue
            
        normal = normal / norm_val
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        
        facet_radius = np.max(angles)
        max_radius = max(max_radius, facet_radius)
    
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Vectorized soft covering radius using smooth soft-min over all points
    and soft-max over grid points. Returns (value, grad) where grad has shape (N, 3).
    """
    M = len(grid_points)
    dots = grid_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min over configuration points for each grid point:
    # soft_min_i = -sigma * log(sum_j exp(-angle_ij / sigma))
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)  # (M, 1)
    soft_min = -sigma * (max_neg.squeeze(1) + np.log(sum_exp_neg.squeeze(1)))  # (M,)
    
    # Weights for soft-min: w_ij = exp_neg_ij / sum_exp_neg_i
    w = exp_neg / sum_exp_neg  # (M, N)
    
    # Soft-max over grid points:
    scaled_sm = soft_min / sigma
    max_sm = np.max(scaled_sm)
    exp_sm = np.exp(scaled_sm - max_sm)
    weights_grid = exp_sm / np.sum(exp_sm)  # (M,)
    soft_max_val = sigma * (max_sm + np.log(np.sum(exp_sm)))
    
    # Gradient computation:
    # d(soft_min_i)/d(p_j) = sum_k w_ik * d(angle_ik)/d(p_j)
    # d(angle_ik)/d(p_j) = 0 if j != k, else -g_i / sqrt(1-dot^2) projected to tangent
    
    # For each (i,j) pair where w_ij > 0:
    # contribution to grad[j] from grid point i:
    #   weights_grid[i] * w_ij * (-g_i / sqrt(1-dots[i,j]^2)) projected onto tangent at p_j
    
    # Efficient: for each j, sum over i of weights_grid[i] * w[i,j] * (-g_i / denom[i,j])
    # then project result onto tangent plane at p_j
    
    dot_safe = np.clip(dots, -1.0 + 1e-9, 1.0 - 1e-9)
    denom = np.sqrt(1.0 - dot_safe * dot_safe)  # (M, N)
    
    # Weighted sum: for each j, grad_raw[j] = sum_i weights_grid[i] * w[i,j] * (-g_i / denom[i,j])
    # = sum_i (weights_grid[i] * w[i,j] / denom[i,j]) * (-g_i)
    
    # combined_weight[i,j] = weights_grid[i] * w[i,j] / denom[i,j]
    combined_w = (weights_grid[:, np.newaxis] * w) / denom  # (M, N)
    
    # grad_raw[j] = -sum_i combined_w[i,j] * g_i
    # = -(g @ combined_w)
    grad_raw = -(grid_points.T @ combined_w)  # (N, 3)
    
    # Project each grad_raw[j] onto tangent plane at points[j]
    dot_gp = np.sum(grad_raw * points, axis=1, keepdims=True)  # (N, 1)
    grad = grad_raw - dot_gp * points  # (N, 3)
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=3):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    Vectorized using bincount for speed.
    """
    for _ in range(n_iters):
        dots = grid_points @ points.T
        nearest = np.argmax(dots, axis=1)  # max dot = min angle
        
        # Compute weighted means using bincount
        new_points = np.zeros((N, 3))
        counts = np.bincount(nearest, minlength=N).astype(float)
        
        for j in range(N):
            if counts[j] > 0:
                new_points[j] = np.sum(grid_points[nearest == j], axis=0) / counts[j]
            else:
                new_points[j] = points[j]
        
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        
        if np.max(diff) < 1e-6:
            break
    
    return points

def construct():
    start_time = time.time()
    
    def _elapsed():
        return time.time() - start_time
    
    # Generate grid on sphere - use fewer points for speed, they're only for the surrogate
    n_grid = 8000
    i = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate starting configurations
    starts = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    spiral_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(spiral_points)
    
    # Random points
    for _ in range(8):
        rp = rng.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # Process each start
    for idx, start_pts in enumerate(starts):
        if _elapsed() > DEADLINE - 20.0:
            break
        
        points = start_pts.copy()
        
        # Step 1: Lloyd's iterations to get a good initial spread
        if _elapsed() < DEADLINE - 18.0:
            for _ in range(6):
                if _elapsed() > DEADLINE - 16.0:
                    break
                points = _lloyd_step(points, grid_points, n_iters=2)
        
        # Step 2: Multi-stage gradient optimization with sigma annealing
        sigmas = [0.15, 0.08, 0.04, 0.02, 0.01]
        
        for sigma in sigmas:
            if _elapsed() > DEADLINE - 10.0:
                break
            
            def objective_and_grad(x_flat, sigma=sigma):
                pts = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts = pts / norms
                
                val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=sigma)
                return val, grad.flatten()
            
            x0 = points.flatten()
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
                )
                if res.x is not None:
                    points = res.x.reshape(N, 3)
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    points = points / norms
            except Exception:
                pass
        
        # Step 3: Final polish with very small sigma
        if _elapsed() < DEADLINE - 5.0:
            def objective_and_grad_polish(x_flat):
                pts = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts = pts / norms
                
                val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.005)
                return val, grad.flatten()
            
            try:
                res = minimize(
                    objective_and_grad_polish,
                    points.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res.x is not None:
                    points = res.x.reshape(N, 3)
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    points = points / norms
            except Exception:
                pass
        
        # Evaluate exact covering radius
        score = _covering_radius_from_hull(points)
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
        
        if best_score < 0.1975:
            break
    
    if best_points is None:
        best_points = spiral_points
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
