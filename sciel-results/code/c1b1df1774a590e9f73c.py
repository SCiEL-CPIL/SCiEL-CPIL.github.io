import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

# EVOLVE-BLOCK-START

def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    return v / norms[:, np.newaxis]

def _covering_radius_from_hull(points):
    """
    Compute the exact covering radius (max circumradius of convex hull facets) in radians.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
        
    max_radius = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex]
        
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
            
        c = n / norm_n
        
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        cos_angle = np.dot(c, v1)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if angle > max_radius:
            max_radius = angle
            
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Vectorized soft covering radius and its gradient using log-sum-exp for stability.
    """
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min with log-sum-exp
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)
    exp_neg = np.exp(neg_scaled - max_neg)
    log_sum_neg = max_neg.squeeze(1) + np.log(np.sum(exp_neg, axis=1))  # (M,)
    log_weights = neg_scaled - log_sum_neg[:, np.newaxis]  # (M, N)
    weights = np.exp(log_weights)  # (M, N)
    
    soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
    
    # Soft-max with log-sum-exp
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    log_sum_max = max_scaled + np.log(np.sum(exp_max))
    log_max_weights = scaled_max - log_sum_max
    max_weights = np.exp(log_max_weights)  # (M,)
    soft_max_val = sigma * log_sum_max
    
    # Gradient
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))
    coeff = max_weights[:, np.newaxis] * weights
    alpha = coeff / denom
    
    term1 = -(alpha.T @ grid_points)
    beta = alpha * dots
    beta_sum = np.sum(beta, axis=0)
    term2 = beta_sum[:, np.newaxis] * points
    
    grad = term1 + term2
    
    return soft_max_val, grad

def _generate_grid(n_theta=60, n_phi=120):
    """Generate a grid of points on the sphere."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

def _lloyd_step(points):
    """
    Lloyd-style step: for each point, move it toward the centroid of its
    Voronoi cell (approximated by the nearest grid points).
    """
    grid_points = _generate_grid(50, 100)
    dots = grid_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # For each grid point, find nearest config point
    nearest = np.argmin(angles, axis=1)  # (M,)
    
    new_points = np.zeros_like(points)
    counts = np.zeros(N)
    
    for i in range(len(grid_points)):
        j = nearest[i]
        new_points[j] += grid_points[i]
        counts[j] += 1
    
    # Where count > 0, move toward the centroid of assigned grid points
    for j in range(N):
        if counts[j] > 0:
            centroid = new_points[j] / counts[j]
            norm_c = np.linalg.norm(centroid)
            if norm_c > 1e-10:
                points[j] = _normalize(centroid)
            # else: keep original point
    
    return points

def _geometric_polish(points, max_iter=2000, deadline=145.0):
    """
    Geometric polish: iteratively find worst facet and adjust vertices.
    Uses adaptive step size.
    """
    best_points = points.copy()
    best_radius = _covering_radius_from_hull(best_points)
    
    alpha = 0.15
    no_improve_count = 0
    patience = 100
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        max_radius = 0.0
        worst_simplex = None
        worst_center = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex]
            n = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            c = n / norm_n
            
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            cos_angle = np.dot(c, v1)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            if angle > max_radius:
                max_radius = angle
                worst_simplex = simplex
                worst_center = c
                
        if worst_simplex is None:
            break
            
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > patience:
            # Reset to best and try smaller steps
            points = best_points.copy()
            alpha *= 0.7
            no_improve_count = 0
            if alpha < 0.01:
                break
                
        if max_radius < 11.3165625 * np.pi / 180.0 + 1e-6:
            break
            
        for idx in worst_simplex:
            points[idx] = (1.0 - alpha) * points[idx] + alpha * worst_center
            points[idx] = _normalize(points[idx])
        
        # Periodic noise to escape local minima
        if it % 100 == 50:
            noise = np.random.normal(0, 0.005, points.shape)
            points = _normalize(points + noise)
    
    return best_points

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    # 2. Gradient-based optimization with adaptive sigma
    grid_points = _generate_grid(50, 100)
    
    # Phase 1: Coarse optimization with larger sigma
    sigma1 = 0.08
    
    def objective_and_grad_coarse(x):
        pts = _normalize(x.reshape(-1, 3))
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=sigma1)
        return val, grad.ravel()
    
    try:
        res = minimize(
            objective_and_grad_coarse,
            points.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        points = _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass
    
    # Lloyd step to redistribute
    if time.time() < deadline - 30:
        points = _lloyd_step(points)
    
    # Phase 2: Fine optimization with smaller sigma
    sigma2 = 0.03
    
    def objective_and_grad_fine(x):
        pts = _normalize(x.reshape(-1, 3))
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=sigma2)
        return val, grad.ravel()
    
    try:
        res = minimize(
            objective_and_grad_fine,
            points.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        points = _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass
    
    # 3. Geometric polish
    if time.time() < deadline - 5:
        points = _geometric_polish(points, max_iter=3000, deadline=deadline - 2)
    
    # Final normalization
    points = _normalize(points)
    
    return points

# EVOLVE-BLOCK-END
