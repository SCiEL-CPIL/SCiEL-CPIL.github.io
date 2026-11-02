import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130
DEADLINE = 145.0  # safety margin before 150s kill

def _covering_radius(points):
    """
    Compute the exact covering radius from the convex hull.
    For each facet (triangle) of the convex hull, compute the circumradius
    of the spherical triangle. The covering radius is the maximum.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    n = len(points)
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        v1 = b - a
        v2 = c - a
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        
        if norm_len < 1e-12:
            continue
        
        normal /= norm_len
        
        # Orient outward: check if centroid is on the same side
        centroid = (a + b + c) / 3.0
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # Angular distances from cap center (normal) to vertices
        dots = np.clip(np.dot(tri, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        
        facet_radius = np.max(angles)
        
        if facet_radius > max_radius:
            max_radius = facet_radius
    
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.03):
    """
    Vectorized soft covering radius and its gradient using smooth soft-min over all points.
    Uses squared angular distances for better conditioning.
    Returns (value, grad) where grad has shape (N, 3).
    """
    # Compute angular distances: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    sq_dists = angles * angles  # (M, N)
    
    # Soft-min of squared distances: for each grid point, weighted average of sq_dists
    # w_j = exp(-sq_dist_j / sigma) / sum_k exp(-sq_dist_k / sigma)
    neg_scaled = -sq_dists / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
    
    # Soft-min value: weighted sum of squared distances
    soft_min_sq = np.sum(weights * sq_dists, axis=1)  # (M,)
    
    # Soft-max of soft-min values (to get covering radius approximation)
    scaled_max = soft_min_sq / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_sq_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    # The soft covering radius (in angle units) is sqrt of soft_max_sq_val
    soft_max_val = np.sqrt(max(soft_max_sq_val, 1e-15))
    
    # Gradient computation for the soft covering radius:
    # f = sqrt(g) where g = soft_max_sq_val
    # df/dg = 1/(2*sqrt(g))
    # dg/d(soft_min_sq_i) = max_weights[i]
    # d(soft_min_sq_i)/d(points_j) = sum_k weights[i,k] * d(sq_dist_ik)/d(points_j)
    # d(sq_dist_ik)/d(points_k) = 2 * angle_ik * d(angle_ik)/d(points_k)
    # d(angle_ik)/d(points_k) = -g_i/sqrt(1-dots^2) + dots/sqrt(1-dots^2) * p_k (tangent)
    
    # Let's compute d(soft_min_sq_i)/d(points_j) for each i, j:
    # For j != k: contribution is 0
    # For j = k: 2 * angle_ik * (-g_i/denom + dots/denom * p_k)
    # But we need the full sum over k for each i, then project to j
    
    # Actually, let's be more careful:
    # soft_min_sq_i = sum_k w_ik * (angle_ik)^2
    # d(soft_min_sq_i)/d(p_j) = sum_k w_ik * 2*angle_ik * d(angle_ik)/d(p_j)
    # d(angle_ik)/d(p_j) = 0 if j != k
    # = -g_i/sqrt(1-dots_ik^2) + dots_ik/sqrt(1-dots_ik^2) * p_k if j = k
    
    # So d(soft_min_sq_i)/d(p_j) = w_ij * 2*angle_ij * (-g_i/denom_ij + dots_ij/denom_ij * p_j)
    
    # Then dg/d(p_j) = sum_i max_weights[i] * d(soft_min_sq_i)/d(p_j)
    
    # df/d(p_j) = (1/(2*sqrt(g))) * dg/d(p_j)
    
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    
    # coeff[i,j] = max_weights[i] * weights[i,j] * 2 * angles[i,j] / denom[i,j]
    coeff = max_weights[:, np.newaxis] * weights * (2.0 * angles / denom)  # (M, N)
    
    # Term 1: -sum_i coeff[i,j] * g_i
    term1 = -(coeff.T @ grid_points)  # (N, 3)
    
    # Term 2: sum_i coeff[i,j] * dots[i,j] * p_j
    beta = coeff * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
    
    dg_dp = term1 + term2  # (N, 3)
    
    # Scale by 1/(2*sqrt(g))
    scale_factor = 1.0 / (2.0 * np.sqrt(max(soft_max_sq_val, 1e-15)))
    grad = scale_factor * dg_dp  # (N, 3)
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=3, time_limit=10.0):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    Uses vectorized operations for speed.
    """
    start_time = time.time()
    for _ in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        dots = grid_points @ points.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        nearest = np.argmin(angles, axis=1)
        
        new_points = np.zeros_like(points)
        
        # Vectorized: sum grid points per cluster, then divide by count
        for j in range(N):
            mask = (nearest == j)
            count = np.sum(mask)
            if count > 0:
                new_points[j] = np.sum(grid_points[mask], axis=0) / count
            else:
                new_points[j] = points[j]
        
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        
        if np.max(diff) < 1e-7:
            break
    
    return points

def _fibonacci_spiral(n):
    """Generate points on a Fibonacci spiral on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_with_lbfgsb(points, grid_points, time_limit=10.0):
    """
    Optimize using L-BFGS-B on the soft covering radius objective.
    Flattens the point cloud, optimizes, then re-normalizes.
    """
    start_time = time.time()
    n = len(points)
    x0 = points.flatten()
    
    def objective_and_grad(x):
        if time.time() - start_time > time_limit:
            return 0.0, np.zeros_like(x)
        
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.03)
        
        # Project gradient to tangent space for each point
        for i in range(n):
            radial = np.dot(grad[i], pts[i])
            grad[i] -= radial * pts[i]
        
        return val, grad.flatten()
    
    result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
    
    final_pts = result.x.reshape(n, 3)
    final_pts /= np.linalg.norm(final_pts, axis=1, keepdims=True)
    
    return final_pts

def _gradient_descent_covering(points, grid_points, time_limit=10.0):
    """
    Gradient descent on the covering radius using a dense grid.
    """
    start_time = time.time()
    current = points.copy()
    n = len(current)
    
    lr = 0.005
    
    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Compute covering radius and gradient
        # For each grid point, find nearest configuration point
        dot = grid_points @ current.T  # (n_grid, n)
        dot = np.clip(dot, -1.0, 1.0)
        dists = np.arccos(dot)  # (n_grid, n)
        
        # Nearest point for each grid point
        min_dists = np.min(dists, axis=1)  # (n_grid,)
        nearest = np.argmin(dists, axis=1)  # (n_grid,)
        
        # Covering radius is the max of min_dists
        # Gradient: for each configuration point i, the gradient is the sum
        # of contributions from grid points for which i is the nearest
        
        forces = np.zeros((n, 3))
        for k in range(len(grid_points)):
            i_nearest = nearest[k]
            d = min_dists[k]
            
            if d > 0.1:  # Only contribute if not too close
                # The grid point wants to be closer to i_nearest
                # Force on i_nearest: pull it toward the grid point
                # Direction: from i_nearest to grid_point, projected to tangent
                diff = grid_points[k] - current[i_nearest]
                # Project to tangent plane at current[i_nearest]
                radial = np.dot(diff, current[i_nearest])
                tangent = diff - radial * current[i_nearest]
                forces[i_nearest] += tangent / (d + 1e-6)
        
        # Update
        current = current + lr * forces
        current /= np.linalg.norm(current, axis=1, keepdims=True)
        
        # Decay learning rate
        lr *= 0.99
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = np.inf
    best_points = None
    
    # Generate multiple starting configurations
    rng = np.random.default_rng(12345)
    
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Several random starts
    for seed in range(5):
        rng_local = np.random.default_rng(seed * 1000 + 42)
        rp = rng_local.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # 3. Icosahedron-based
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_expanded = []
    rng_ico = np.random.default_rng(777)
    for v in ico_verts:
        for _ in range(11):
            jittered = v + rng_ico.normal(scale=0.15, size=3)
            jittered /= np.linalg.norm(jittered)
            ico_expanded.append(jittered)
    starts.append(np.array(ico_expanded[:N]))
    
    # Create a high-resolution grid for soft optimization
    n_grid = 20000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.clip(1.0 - z_grid * z_grid, 0, 1))
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), 
                                    r_grid * np.sin(phi_grid), 
                                    z_grid])
    
    # Evaluate each start
    for idx, start_pts in enumerate(starts):
        remaining = deadline - time.time()
        
        if remaining < 5.0:
            break
        
        # Evaluate initial covering radius
        score = _covering_radius(start_pts)
        if score < best_score:
            best_score = score
            best_points = start_pts.copy()
        
        # Run Lloyd's algorithm (grid-based)
        lloyd_time = min(15.0, max(5.0, remaining * 0.25))
        lloyd_points = _lloyd_step(start_pts, grid_points, n_iters=10, time_limit=lloyd_time)
        
        score = _covering_radius(lloyd_points)
        if score < best_score:
            best_score = score
            best_points = lloyd_points.copy()
        
        # Further refine with L-BFGS-B on soft objective
        remaining = deadline - time.time()
        if remaining > 5.0:
            lbfgs_time = min(20.0, remaining * 0.4)
            lbfgs_points = _optimize_with_lbfgsb(lloyd_points, grid_points, time_limit=lbfgs_time)
            
            score = _covering_radius(lbfgs_points)
            if score < best_score:
                best_score = score
                best_points = lbfgs_points.copy()
        
        # Final Lloyd polish
        remaining = deadline - time.time()
        if remaining > 3.0:
            polish_time = min(10.0, remaining * 0.3)
            polish_points = _lloyd_step(best_points, grid_points, n_iters=5, time_limit=polish_time)
            score = _covering_radius(polish_points)
            if score < best_score:
                best_score = score
                best_points = polish_points.copy()
    
    # If we have time, do a final L-BFGS-B polish on the best points
    remaining = deadline - time.time()
    if remaining > 10.0 and best_points is not None:
        final_lbfgs_time = min(30.0, remaining * 0.5)
        final_points = _optimize_with_lbfgsb(best_points, grid_points, time_limit=final_lbfgs_time)
        score = _covering_radius(final_points)
        if score < best_score:
            best_score = score
            best_points = final_points.copy()
    
    # Fallback
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Ensure all points are normalized
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
