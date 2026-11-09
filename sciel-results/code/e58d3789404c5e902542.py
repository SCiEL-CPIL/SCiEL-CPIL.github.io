import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Compute areas of all C(N,3) triangles. Returns (areas, min_area, min_idx)."""
    n = len(points)
    # Precompute indices
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_idx = np.argmin(areas)
    return areas, areas[min_idx], (i_flat[min_idx], j_flat[min_idx], k_flat[min_idx])

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points)
    if points.shape != (N, 2):
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    _, min_area, _ = _all_triangle_areas(points)
    
    return min_area / hull_area

def _soft_min_triangle_area(points, temperature=0.05):
    """
    Smooth surrogate for min triangle area using soft-min.
    Lower is better.
    """
    areas, _, _ = _all_triangle_areas(points)
    # Soft min: -T * log(sum(exp(-a/T)))
    # For numerical stability, subtract min first
    a_min = np.min(areas)
    # If all areas are 0, return 0
    if a_min < 1e-15:
        return 0.0
    # exp(-(a - a_min)/T)
    exponents = - (areas - a_min) / temperature
    # Clip to avoid underflow/overflow issues in sum
    # max exponent is 0. min exponent is large negative.
    sum_exp = np.sum(np.exp(exponents))
    soft_min = a_min - temperature * np.log(sum_exp)
    return soft_min

def _objective(points_flat, temperature=0.05):
    """
    Objective function for minimization.
    Maximize min_area / hull_area.
    Equivalent to minimize - (min_area / hull_area).
    Using soft min for smoothness.
    """
    points = points_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10 # Large penalty
    
    soft_min_area = _soft_min_triangle_area(points, temperature)
    
    return - (soft_min_area / hull_area)

def _subgradient_step(points, step_size, temperature=0.05):
    """
    Perform a subgradient ascent step.
    Maximize f(x) = min_area / hull_area.
    Gradient of -f is -grad(f).
    We want to increase f, so we move in direction of grad(f).
    
    f = A_min / H
    df/dp_i = (1/H) * dA_min/dp_i - (A_min/H^2) * dH/dp_i
    
    This is complex. Let's use a simpler heuristic or finite differences for the critical triangle.
    Actually, let's stick to Nelder-Mead but with a better objective or better restarts.
    The subgradient of min area is the gradient of the min area triangle w.r.t its vertices.
    The gradient of hull area is more complex (depends on hull vertices).
    
    Let's try a simpler approach: 
    1. Identify the minimum area triangle (i, j, k).
    2. Compute the gradient of its area w.r.t p_i, p_j, p_k.
    3. Move p_i, p_j, p_k to increase the area.
    4. Also, we should consider the hull area constraint. 
       If we push points out, hull area increases, which lowers the score.
       So we want to push points in a way that increases the min area faster than the hull area increases.
       
    For a triangle with vertices A, B, C, Area = 0.5 * | (B-A) x (C-A) |.
    Let v1 = B-A, v2 = C-A.
    dArea/dA = -0.5 * (v1 x v2) * ... wait, vector calculus.
    Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|.
    
    Let's use finite differences for the subgradient step to be safe and robust.
    """
    # This might be too slow or unstable. Let's rely on Nelder-Mead with multiple starts.
    # But we can add a "push apart" heuristic for the min triangle.
    
    areas, min_area, min_idx = _all_triangle_areas(points)
    if min_area < 1e-12:
        return points
    
    i, j, k = min_idx
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    # Gradient of area of triangle (i,j,k) w.r.t p_i, p_j, p_k
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # Let cross = (p_j - p_i) x (p_k - p_i)
    # dArea/dp_i = -0.5 * ( (p_j-p_i) x (p_k-p_i) ) / |cross| * (p_j - p_k) ?? 
    # Actually, for 2D:
    # A = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # dA/dx_i = 0.5 * (y_j - y_k)
    # dA/dy_i = 0.5 * (x_k - x_j)
    # dA/dx_j = 0.5 * (y_k - y_i)
    # dA/dy_j = 0.5 * (x_i - x_k)
    # dA/dx_k = 0.5 * (y_i - y_j)
    # dA/dy_k = 0.5 * (x_j - x_i)
    
    # These are gradients of the *signed* area (without abs).
    # Since we want to increase the absolute area, we move in the direction of the gradient if the area is positive, or opposite if negative.
    # Or simply, the gradient of |A| is sign(A) * grad(A).
    
    cross_val = (p_j[0] - p_i[0]) * (p_k[1] - p_i[1]) - (p_j[1] - p_i[1]) * (p_k[0] - p_i[0])
    sign = np.sign(cross_val)
    if sign == 0:
        return points
        
    grad_i = np.array([0.5 * (p_j[1] - p_k[1]), 0.5 * (p_k[0] - p_j[0])])
    grad_j = np.array([0.5 * (p_k[1] - p_i[1]), 0.5 * (p_i[0] - p_k[0])])
    grad_k = np.array([0.5 * (p_i[1] - p_j[1]), 0.5 * (p_j[0] - p_i[0])])
    
    # Scale by sign to increase absolute area
    grad_i *= sign
    grad_j *= sign
    grad_k *= sign
    
    # Normalize gradients to prevent large steps
    norm_i = np.linalg.norm(grad_i)
    norm_j = np.linalg.norm(grad_j)
    norm_k = np.linalg.norm(grad_k)
    if norm_i > 1e-12: grad_i /= norm_i
    if norm_j > 1e-12: grad_j /= norm_j
    if norm_k > 1e-12: grad_k /= norm_k
    
    # We also need to account for hull area increase.
    # If i, j, k are on the hull, moving them out increases hull area.
    # A simple heuristic: dampen the step if the point is on the hull.
    # But identifying hull points is expensive.
    # Let's just take a small step.
    
    new_points = points.copy()
    # Adaptive step size
    # If area is small, we need a larger step.
    # step_size is provided.
    new_points[i] += step_size * grad_i
    new_points[j] += step_size * grad_j
    new_points[k] += step_size * grad_k
    
    return new_points

def _optimize_from(x0, max_iter=5000, time_limit=10.0):
    """
    Run Nelder-Mead optimization from x0.
    """
    start_time = time.time()
    
    def objective_wrapper(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _objective(x)
        
    try:
        res = minimize(
            objective_wrapper, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except TimeoutError:
        return x0
    except Exception:
        return x0

def _generate_regular_polygon_configs():
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    configs.append(np.vstack([pts_12, center]))
    
    # 3. Regular 11-gon + 2 inner points
    angles_11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    pts_11 = np.column_stack([np.cos(angles_11), np.sin(angles_11)])
    inner_1 = np.array([[0.5, 0.0]])
    inner_2 = np.array([[-0.5, 0.0]])
    configs.append(np.vstack([pts_11, inner_1, inner_2]))
    
    # 4. Jittered Grid
    rng = np.random.default_rng(42)
    grid = np.array([(i, j) for i in range(4) for j in range(4)])
    grid = grid[:13]
    grid = grid + rng.normal(0, 0.1, grid.shape)
    configs.append(grid)
    
    # 5. Random points in unit circle
    rng = np.random.default_rng(123)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    configs.append(pts)
    
    # 6. Two concentric polygons
    # 7 outer, 6 inner
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    pts_in = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    configs.append(np.vstack([pts_out, pts_in]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    configs = _generate_regular_polygon_configs()
    best_score = -np.inf
    best_points = None
    
    # First pass: Optimize from each config
    for init_points in configs:
        if time.time() > deadline - 20.0:
            break
            
        x0 = init_points.flatten()
        opt_points = _optimize_from(x0, max_iter=2000, time_limit=15.0)
        
        if opt_points is not None:
            candidate_points = opt_points.reshape(N, 2)
            score = get_score(candidate_points)
            
            if score > best_score:
                best_score = score
                best_points = candidate_points
                
        # Check time
        if time.time() > deadline - 10.0:
            break
            
    # Second pass: If we have time, do more restarts with jittered best points
    if best_points is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(999)
        for _ in range(5): # Try a few more
            if time.time() > deadline - 5.0:
                break
            # Jitter the best points
            jittered = best_points + rng.normal(0, 0.05, best_points.shape)
            x0 = jittered.flatten()
            opt_points = _optimize_from(x0, max_iter=1000, time_limit=5.0)
            
            if opt_points is not None:
                candidate_points = opt_points.reshape(N, 2)
                score = get_score(candidate_points)
                
                if score > best_score:
                    best_score = score
                    best_points = candidate_points

    if best_points is None:
        # Fallback
        best_points = configs[0]
        
    return best_points
