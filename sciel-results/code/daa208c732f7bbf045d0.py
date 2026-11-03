"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _exact_covering_radius(points):
    """
    Compute the exact covering radius as the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        # Ensure points are normalized
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex]
            # Normal to the plane containing p1, p2, p3
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            o = n / norm_n
            # Ensure o points in the same direction as the centroid
            centroid = (p1 + p2 + p3) / 3.0
            if np.dot(o, centroid) < 0:
                o = -o
            # Circumradius is the angular distance from o to any vertex
            cos_r = np.dot(o, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        return np.pi

def _surrogate_loss_and_grad(v, beta, Y):
    """
    Compute the smooth surrogate for the covering radius and its gradient.
    v: flattened array of N points (3N values)
    beta: sharpness parameter
    Y: grid of M points on the sphere
    """
    X = v.reshape(-1, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    X = X / norms
    
    # Compute angular distances D (M x N)
    dots = Y @ X.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Softmin over sites j for each grid point i
    # v_i = -1/beta * log(sum_j exp(-beta * dists_ij))
    # Stable version:
    min_dists = np.min(dists, axis=1, keepdims=True)
    exp_terms = np.exp(-beta * (dists - min_dists))
    sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
    v_soft = -(min_dists + np.log(sum_exp)) / beta  # M x 1
    
    # Softmax over grid points i
    # F = 1/beta * log(sum_i exp(beta * v_i))
    # Stable version:
    max_v = np.max(v_soft.flatten())
    exp_v = np.exp(beta * (v_soft.flatten() - max_v))
    sum_exp_v = np.sum(exp_v)
    F = (max_v + np.log(sum_exp_v)) / beta
    
    # Weights p_i
    p = exp_v / sum_exp_v  # M
    
    # Weights q_ij
    q = exp_terms / sum_exp  # M x N
    
    # Gradient calculation
    # grad_X_j = sum_i p_i * q_ij * grad_dists_ij
    # grad_dists_ij = (X_j * (Y_i . X_j) - Y_i) / sin(dists_ij)
    
    sin_dists = np.sqrt(1.0 - dots**2)
    sin_dists[sin_dists < 1e-8] = 1e-8
    
    X_b = X[np.newaxis, :, :]       # 1, N, 3
    Y_b = Y[:, np.newaxis, :]       # M, 1, 3
    dots_b = dots[:, :, np.newaxis] # M, N, 1
    sin_b = sin_dists[:, :, np.newaxis] # M, N, 1
    p_b = p[:, np.newaxis, np.newaxis]  # M, 1, 1
    q_b = q[:, :, np.newaxis]           # M, N, 1
    
    term = (X_b * dots_b - Y_b) / sin_b
    weights = p_b * q_b
    
    grad_X = np.sum(weights * term, axis=0)  # N, 3
    
    # Project to ambient space
    grad_v = grad_X / norms.flatten()
    
    return F, grad_v.flatten()

def construct():
    global START_TIME, TIME_LIMIT
    
    # Generate a dense grid for the surrogate
    M = 20000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    Y = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    configs = []
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random configurations
    np.random.seed(42)
    for _ in range(6):
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
        
    # Optimization stages
    betas = [20, 50, 100, 200, 400]
    
    candidates = []
    
    for start_idx, init_pts in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        v = init_pts.flatten()
        current_points = init_pts.copy()
        
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            try:
                res = minimize(
                    _surrogate_loss_and_grad,
                    v,
                    args=(beta, Y),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                v = res.x
                current_points = v.reshape(-1, 3)
                current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            except Exception:
                pass
        
        # Evaluate exact radius for this candidate
        rad = _exact_covering_radius(current_points)
        candidates.append((rad, current_points))
        
        if rad < best_radius:
            best_radius = rad
            best_points = current_points.copy()
            
    # Polish the best candidate with higher beta
    if best_points is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        v = best_points.flatten()
        for beta in [800, 1200]:
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
            try:
                res = minimize(
                    _surrogate_loss_and_grad,
                    v,
                    args=(beta, Y),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                v = res.x
                best_points = v.reshape(-1, 3)
                best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
            except Exception:
                break
        
        # Final check
        final_rad = _exact_covering_radius(best_points)
        if final_rad < best_radius:
            best_radius = final_rad

    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_points

# EVOLVE-BLOCK-END
