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
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute the normal to the facet
        v1 = b - a
        v2 = c - a
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        
        if norm_len < 1e-12:
            continue
        
        normal /= norm_len
        
        # Ensure normal points outward (away from origin)
        centroid = np.mean(tri, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # Compute angular distances from normal (cap center) to vertices
        d0 = np.arccos(np.clip(np.dot(normal, a), -1, 1))
        d1 = np.arccos(np.clip(np.dot(normal, b), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, c), -1, 1))
        
        # If the cap is very large (> 90 degrees), the center might be on the other side
        if max(d0, d1, d2) > np.pi / 2:
            d0 = np.arccos(np.clip(np.dot(-normal, a), -1, 1))
            d1 = np.arccos(np.clip(np.dot(-normal, b), -1, 1))
            d2 = np.arccos(np.clip(np.dot(-normal, c), -1, 1))
        
        cap_radius = max(d0, d1, d2)
        
        if cap_radius > max_radius:
            max_radius = cap_radius
    
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

def _lloyd_iteration(points):
    """
    One Lloyd iteration: for each point, compute the centroid of its Voronoi cell
    on the sphere and move the point there.
    Uses a dense grid to approximate the Voronoi cells.
    """
    # Create a dense grid of test points on the sphere
    n_grid = 10000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.clip(1.0 - z_grid * z_grid, 0, 1))
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), 
                                    r_grid * np.sin(phi_grid), 
                                    z_grid])
    
    # For each grid point, find nearest configuration point
    dots = grid_points @ points.T
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    nearest = np.argmin(angles, axis=1)
    
    # Compute new positions as centroid of Voronoi cell
    new_points = np.zeros_like(points)
    counts = np.zeros(N)
    
    for j in range(N):
        mask = (nearest == j)
        count = np.sum(mask)
        if count > 0:
            new_points[j] = np.sum(grid_points[mask], axis=0) / count
        else:
            new_points[j] = points[j]
        counts[j] = count
    
    # Normalize
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    new_points = new_points / norms
    
    return new_points

def _lloyd_optimize(points, max_iter=50, tol=1e-8):
    """
    Run Lloyd's algorithm for spherical k-means covering.
    """
    current = points.copy()
    
    for it in range(max_iter):
        new_points = _lloyd_iteration(current)
        
        # Check convergence
        diff = np.linalg.norm(new_points - current)
        current = new_points
        
        if diff < tol:
            break
    
    return current

def _gradient_descent_covering(points, grid_points, time_limit=10.0, sigma=0.05):
    """
    Gradient descent on the soft covering radius using a dense grid.
    """
    start_time = time.time()
    current = points.copy()
    n = len(current)
    
    lr = 0.003
    
    for it in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Compute soft covering radius and gradient
        val, grad = _soft_covering_radius_and_grad(current, grid_points, sigma=sigma)
        
        # Update
        current = current + lr * grad
        current /= np.linalg.norm(current, axis=1, keepdims=True)
        
        # Decay learning rate
        lr *= 0.995
        
        # Adaptively adjust sigma: decrease it over time for sharper gradients
        sigma = max(0.01, sigma * 0.999)
    
    return current

def _fibonacci_spiral(n):
    """Generate points on a Fibonacci spiral on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _create_grid(n_grid=20000):
    """Create a dense grid of points on the sphere for optimization."""
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.clip(1.0 - z_grid * z_grid, 0, 1))
    return np.column_stack([r_grid * np.cos(phi_grid), 
                            r_grid * np.sin(phi_grid), 
                            z_grid])

def _local_polish(points, time_limit=5.0):
    """
    High-precision local polish: iteratively refine the configuration by
    targeting the worst facets of the convex hull.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    for it in range(100):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the worst facet (largest circumradius)
        worst_facet = None
        worst_radius = -1
        worst_normal = None
        
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            
            v1 = b - a
            v2 = c - a
            normal = np.cross(v1, v2)
            norm_val = np.linalg.norm(normal)
            
            if norm_val < 1e-12:
                continue
            
            normal = normal / norm_val
            
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            
            d0 = np.arccos(np.clip(np.dot(normal, a), -1, 1))
            d1 = np.arccos(np.clip(np.dot(normal, b), -1, 1))
            d2 = np.arccos(np.clip(np.dot(normal, c), -1, 1))
            
            if max(d0, d1, d2) > np.pi / 2:
                d0 = np.arccos(np.clip(np.dot(-normal, a), -1, 1))
                d1 = np.arccos(np.clip(np.dot(-normal, b), -1, 1))
                d2 = np.arccos(np.clip(np.dot(-normal, c), -1, 1))
            
            facet_radius = max(d0, d1, d2)
            
            if facet_radius > worst_radius:
                worst_radius = facet_radius
                worst_facet = simplex
                worst_normal = normal
        
        if worst_facet is None:
            break
        
        # Move the vertices of the worst facet slightly toward the cap center
        step_size = 0.001 * (worst_radius - np.pi / 20)  # Adaptive step
        step_size = max(0.0001, min(0.01, step_size))
        
        for idx in worst_facet:
            # Move point toward the normal direction (cap center)
            pts[idx] = pts[idx] + step_size * worst_normal
            pts[idx] /= np.linalg.norm(pts[idx])
        
        # Also do a small Lloyd step
        new_pts = _lloyd_iteration(pts)
        pts = 0.5 * pts + 0.5 * new_pts
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    return pts

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
    for seed in range(3):
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
    
    # Create grid for optimization
    grid_points = _create_grid(20000)
    
    # Evaluate each start with optimization pipeline
    for idx, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        if remaining < 10.0:
            break
        
        # Evaluate initial covering radius
        score = _covering_radius(start_pts)
        if score < best_score:
            best_score = score
            best_points = start_pts.copy()
        
        # Step 1: Lloyd's algorithm for coarse optimization
        lloyd_time = min(15.0, max(5.0, remaining * 0.3))
        lloyd_points = _lloyd_optimize(start_pts, max_iter=30)
        
        score = _covering_radius(lloyd_points)
        if score < best_score:
            best_score = score
            best_points = lloyd_points.copy()
        
        # Step 2: Gradient descent with soft covering radius
        remaining = deadline - time.time()
        if remaining > 10.0:
            gd_time = min(15.0, remaining * 0.4)
            gd_points = _gradient_descent_covering(lloyd_points, grid_points, 
                                                    time_limit=gd_time, sigma=0.05)
            
            score = _covering_radius(gd_points)
            if score < best_score:
                best_score = score
                best_points = gd_points.copy()
        
        # Step 3: Final local polish targeting worst facets
        remaining = deadline - time.time()
        if remaining > 5.0:
            polish_time = min(5.0, remaining * 0.5)
            polish_points = _local_polish(best_points, time_limit=polish_time)
            
            score = _covering_radius(polish_points)
            if score < best_score:
                best_score = score
                best_points = polish_points.copy()
    
    # Final polish on the best result
    remaining = deadline - time.time()
    if remaining > 3.0:
        final_polish_time = min(3.0, remaining * 0.8)
        final_points = _local_polish(best_points, time_limit=final_polish_time)
        
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
