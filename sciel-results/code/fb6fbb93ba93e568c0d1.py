import numpy as np
import time
from scipy.spatial import ConvexHull

N = 13

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """Compute areas of all C(N,3) triangles. Returns (areas, min_idx)."""
    n = len(points)
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
    return areas, (i_flat[min_idx], j_flat[min_idx], k_flat[min_idx])

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
        
    areas, _ = _all_triangle_areas(points)
    
    return np.min(areas) / hull_area

def _subgradient_step(points, step_size=0.1):
    """
    Perform a subgradient ascent step on the minimum-area triangle.
    Returns new points and the critical triangle indices.
    """
    areas, min_idx = _all_triangle_areas(points)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return points, min_idx
    
    i, j, k = min_idx
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    # Compute cross product to determine orientation
    cross_val = (p_j[0] - p_i[0]) * (p_k[1] - p_i[1]) - (p_j[1] - p_i[1]) * (p_k[0] - p_i[0])
    sign = np.sign(cross_val)
    if sign == 0:
        return points, min_idx
    
    # Gradients of signed area w.r.t each vertex
    grad_i = np.array([0.5 * (p_j[1] - p_k[1]), 0.5 * (p_k[0] - p_j[0])])
    grad_j = np.array([0.5 * (p_k[1] - p_i[1]), 0.5 * (p_i[0] - p_k[0])])
    grad_k = np.array([0.5 * (p_i[1] - p_j[1]), 0.5 * (p_j[0] - p_i[0])])
    
    # Scale by sign to increase absolute area
    grad_i *= sign
    grad_j *= sign
    grad_k *= sign
    
    # Normalize to unit vectors
    norm_i = np.linalg.norm(grad_i)
    norm_j = np.linalg.norm(grad_j)
    norm_k = np.linalg.norm(grad_k)
    if norm_i > 1e-12: grad_i /= norm_i
    if norm_j > 1e-12: grad_j /= norm_j
    if norm_k > 1e-12: grad_k /= norm_k
    
    new_points = points.copy()
    new_points[i] += step_size * grad_i
    new_points[j] += step_size * grad_j
    new_points[k] += step_size * grad_k
    
    return new_points, min_idx

def _optimize_subgradient(x0, max_iter=2000, time_limit=20.0):
    """
    Optimize using subgradient method with adaptive step size.
    """
    start_time = time.time()
    points = x0.copy()
    best_score = get_score(points)
    best_points = points.copy()
    
    # Adaptive step size
    step_size = 0.15
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perform subgradient step
        new_points, min_idx = _subgradient_step(points, step_size)
        
        # Evaluate new score
        new_score = get_score(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points
            patience = 0
            # Increase step size slightly when improving
            step_size = min(step_size * 1.05, 0.3)
        else:
            patience += 1
            # Decrease step size when not improving
            step_size = max(step_size * 0.95, 0.01)
            
            if patience > max_patience:
                break
    
    return best_points, best_score

def _generate_initials():
    """Generate diverse initial configurations."""
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
    inner_1 = np.array([[0.4, 0.0]])
    inner_2 = np.array([[-0.4, 0.0]])
    configs.append(np.vstack([pts_11, inner_1, inner_2]))
    
    # 4. Two concentric polygons: 7 outer, 6 inner
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    pts_in = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    configs.append(np.vstack([pts_out, pts_in]))
    
    # 5. Three concentric polygons: 5 outer, 5 middle, 3 inner
    angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pts_5 = np.column_stack([np.cos(angles_5), np.sin(angles_5)])
    angles_5b = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    pts_5b = np.column_stack([0.6 * np.cos(angles_5b), 0.6 * np.sin(angles_5b)])
    angles_3 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    pts_3 = np.column_stack([0.25 * np.cos(angles_3), 0.25 * np.sin(angles_3)])
    configs.append(np.vstack([pts_5, pts_5b, pts_3]))
    
    # 6. Random points in unit disk
    rng = np.random.default_rng(42)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    configs.append(pts)
    
    # 7. Grid-like with jitter
    rng = np.random.default_rng(123)
    grid = np.array([(i, j) for i in range(4) for j in range(4)])[:13]
    grid = (grid - 1.5) / 1.5 + rng.normal(0, 0.08, grid.shape)
    configs.append(grid)
    
    # 8. Regular 13-gon with slight radial perturbation
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = 1.0 + 0.1 * np.sin(2 * angles)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    configs.append(pts)
    
    # 9. Hexagonal lattice pattern
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for a in np.linspace(0, 2*np.pi, 6, endpoint=False):
        pts.append([0.5*np.cos(a), 0.5*np.sin(a)])
    # Ring 2: 6 points
    for a in np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6:
        pts.append([1.0*np.cos(a), 1.0*np.sin(a)])
    configs.append(np.array(pts))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    configs = _generate_initials()
    best_score = -np.inf
    best_points = None
    
    # First pass: Optimize from each config using subgradient
    for init_points in configs:
        if time.time() > deadline - 15.0:
            break
            
        opt_points, score = _optimize_subgradient(init_points, max_iter=3000, time_limit=12.0)
        
        if score > best_score:
            best_score = score
            best_points = opt_points
    
    # Second pass: If we have time, do more restarts with jittered best points
    if best_points is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(999)
        for _ in range(8):
            if time.time() > deadline - 5.0:
                break
            # Jitter the best points
            jittered = best_points + rng.normal(0, 0.03, best_points.shape)
            opt_points, score = _optimize_subgradient(jittered, max_iter=2000, time_limit=6.0)
            
            if score > best_score:
                best_score = score
                best_points = opt_points
    
    # Third pass: Fine-tune the best solution with smaller steps
    if best_points is not None and time.time() < deadline - 3.0:
        # Use very small steps for final refinement
        points = best_points
        step_size = 0.02
        for _ in range(500):
            if time.time() > deadline - 1.0:
                break
            new_points, _ = _subgradient_step(points, step_size)
            new_score = get_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                points = new_points
            else:
                step_size *= 0.9
                if step_size < 0.001:
                    break
    
    if best_points is None:
        best_points = configs[0]
        
    return best_points
