import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130

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

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Vectorized soft covering radius and its gradient using smooth soft-min over all points.
    For each grid point, we compute a weighted sum over all configuration points with
    weights from a soft-min of the angular distances. This gives a smooth gradient.
    Returns (value, grad) where grad has shape (N, 3).
    """
    # Compute angular distances: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min: for each grid point, compute weighted average of angles
    # using softmax weights over the negative angles (to get min)
    # w_j = exp(-angle_j / sigma) / sum_k exp(-angle_k / sigma)
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
    
    # Soft-min value: weighted sum of angles
    soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
    
    # Soft-max of soft-min values (to get covering radius approximation)
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    # Gradient computation:
    # d(soft_min_i)/d(points_j) = sum_k weights_ik * d(angle_ik)/d(points_j)
    # d(angle_ik)/d(points_j) = 0 if j != k, else -g_i / sqrt(1 - dot^2) projected to tangent
    
    # For each grid point i and config point j:
    # contribution = max_weights[i] * weights[i,j] * (-g_i / sqrt(1 - dots[i,j]^2)) * tangent_projection
    
    # Compute per (i,j) gradient contributions
    # denom: (M, N)
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    
    # Raw gradient for each (i, j): -g_i / denom[i,j]
    # We need: for each j, sum_i max_weights[i] * weights[i,j] * (-g_i / denom[i,j]) * tangent_proj
    
    # tangent projection: grad_raw - (grad_raw . p_j) * p_j
    # grad_raw[i,j] = -g_i / denom[i,j]
    # grad_raw . p_j = -dot(g_i, p_j) / denom[i,j] = -dots[i,j] / denom[i,j]
    
    # So tangent_grad[i,j] = -g_i / denom[i,j] + dots[i,j]/denom[i,j] * p_j
    
    # Weighted contribution to point j from grid point i:
    # coeff[i,j] = max_weights[i] * weights[i,j]
    # grad_j = sum_i coeff[i,j] * (-g_i / denom[i,j] + dots[i,j]/denom[i,j] * p_j)
    
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)
    
    # Term 1: sum_i coeff[i,j] * (-g_i / denom[i,j])
    # = -sum_i (coeff[i,j] / denom[i,j]) * g_i
    # Let alpha[i,j] = coeff[i,j] / denom[i,j]
    alpha = coeff / denom  # (M, N)
    
    # Term1_j = -sum_i alpha[i,j] * g_i
    # This is: -(alpha.T @ grid_points)  -> shape (N, 3)
    term1 = -(alpha.T @ grid_points)  # (N, 3)
    
    # Term 2: sum_i coeff[i,j] * dots[i,j] / denom[i,j] * p_j
    # = sum_i alpha[i,j] * dots[i,j] * p_j
    # For each j: p_j * sum_i alpha[i,j] * dots[i,j]
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
    
    grad = term1 + term2  # (N, 3)
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=3):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    Uses vectorized operations for speed.
    """
    for _ in range(n_iters):
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
        
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.05)
        
        # Project gradient to tangent space for each point
        for i in range(n):
            radial = np.dot(grad[i], pts[i])
            grad[i] -= radial * pts[i]
        
        return val, grad.flatten()
    
    result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    
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
    deadline = start_time + 140.0  # 10s safety margin
    
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
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        if remaining < 5.0:
            break
        
        # Evaluate initial covering radius
        score = _covering_radius(start_pts)
        if score < best_score:
            best_score = score
            best_points = start_pts.copy()
        
        # Run Lloyd's algorithm (grid-based)
        lloyd_time = min(15.0, max(5.0, remaining * 0.3))
        lloyd_points = _lloyd_step(start_pts, grid_points, n_iters=10)
        
        score = _covering_radius(lloyd_points)
        if score < best_score:
            best_score = score
            best_points = lloyd_points.copy()
        
        # Further refine with L-BFGS-B on soft objective
        remaining = deadline - time.time()
        if remaining > 5.0:
            lbfgs_time = min(15.0, remaining * 0.4)
            lbfgs_points = _optimize_with_lbfgsb(lloyd_points, grid_points, time_limit=lbfgs_time)
            
            score = _covering_radius(lbfgs_points)
            if score < best_score:
                best_score = score
                best_points = lbfgs_points.copy()
        
        # Final Lloyd polish
        remaining = deadline - time.time()
        if remaining > 3.0:
            polish_points = _lloyd_step(best_points, grid_points, n_iters=5)
            score = _covering_radius(polish_points)
            if score < best_score:
                best_score = score
                best_points = polish_points.copy()
    
    # Fallback
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Ensure all points are normalized
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
