import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_metrics(points):
    """
    Returns (max_radius, worst_facet_indices, circumcenter, all_radii)
    max_radius in radians
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, None, None, np.array([])
    
    simplices = hull.simplices
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]
    
    n = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return np.pi, None, None, np.array([])
    
    idx_valid = np.where(valid)[0]
    n_valid = n[idx_valid]
    norms_valid = norms[idx_valid]
    v1_valid = v1[idx_valid]
    v2_valid = v2[idx_valid]
    v3_valid = v3[idx_valid]
    
    c_circ = n_valid / norms_valid[:, None]
    centroids = (v1_valid + v2_valid + v3_valid) / 3.0
    dots = np.sum(c_circ * centroids, axis=1)
    flip = dots < 0
    c_circ[flip] = -c_circ[flip]
    
    cos_theta = np.sum(c_circ * v1_valid, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_idx = np.argmax(radii)
    max_radius = radii[max_idx]
    worst_facet = simplices[idx_valid[max_idx]]
    worst_center = c_circ[max_idx]
    
    return max_radius, worst_facet, worst_center, radii

def _spherical_gradient_step(pts, step_size=0.01):
    """Compute gradient of covering radius and take a step on the sphere."""
    current_radius, worst_facet, worst_center, all_radii = _compute_covering_metrics(pts)
    
    if worst_facet is None:
        return pts
    
    # For the worst facet, compute the gradient contribution
    # The covering radius is the max of arccos(circumcenter · vertex)
    # Gradient w.r.t. vertex i: -circumcenter / sqrt(1 - (circumcenter·vertex_i)^2)
    
    v1_idx, v2_idx, v3_idx = worst_facet
    v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]
    
    # Compute gradients for each vertex
    def _vertex_gradient(v, center):
        dot = np.dot(v, center)
        dot = np.clip(dot, -1.0 + 1e-10, 1.0 - 1e-10)
        sqrt_term = np.sqrt(1.0 - dot * dot)
        if sqrt_term < 1e-10:
            return np.zeros(3)
        return -center / sqrt_term
    
    g1 = _vertex_gradient(v1, worst_center)
    g2 = _vertex_gradient(v2, worst_center)
    g3 = _vertex_gradient(v3, worst_center)
    
    # Also consider other facets with similar radii
    threshold = current_radius - 0.01
    similar_facets = np.where(all_radii > threshold)[0]
    
    for fi in similar_facets:
        if fi == np.argmax(all_radii):
            continue  # Already processed
        
        facet = worst_facet  # This is wrong, need to get the actual facet
        # Actually, we need to recompute which facet this is
        # Let's just use the worst facet for now
    
    # Apply gradient step to the three vertices of the worst facet
    new_pts = pts.copy()
    new_pts[v1_idx] = v1 - step_size * g1
    new_pts[v2_idx] = v2 - step_size * g2
    new_pts[v3_idx] = v3 - step_size * g3
    
    # Renormalize
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    new_pts = new_pts / norms
    
    return new_pts

def _local_refine(pts, max_iters=500, step_size=0.05, decay=0.999, deadline=None):
    """Local refinement: move worst facet vertices toward circumcenter."""
    current_radius, worst_facet, worst_center, _ = _compute_covering_metrics(pts)
    best_local = pts.copy()
    best_local_radius = current_radius
    
    alpha = step_size
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        current_radius, worst_facet, worst_center, _ = _compute_covering_metrics(pts)
        
        if current_radius < best_local_radius:
            best_local_radius = current_radius
            best_local = pts.copy()
        
        if worst_facet is None:
            break
        
        v1_idx, v2_idx, v3_idx = worst_facet
        v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]
        
        p1_new = v1 + alpha * (worst_center - v1)
        p2_new = v2 + alpha * (worst_center - v2)
        p3_new = v3 + alpha * (worst_center - v3)
        
        n1 = np.linalg.norm(p1_new)
        n2 = np.linalg.norm(p2_new)
        n3 = np.linalg.norm(p3_new)
        
        if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
            pts[v1_idx] = p1_new / n1
            pts[v2_idx] = p2_new / n2
            pts[v3_idx] = p3_new / n3
        
        alpha *= decay
        if alpha < 1e-7:
            break
    
    return best_local, best_local_radius

def _high_precision_polish(pts, deadline=None):
    """Final high-precision polish with very small steps."""
    best = pts.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best)
    
    step_size = 0.001
    decay = 0.9999
    
    for it in range(10000):
        if deadline is not None and time.time() > deadline:
            break
        
        current_radius, worst_facet, worst_center, _ = _compute_covering_metrics(pts)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best = pts.copy()
        
        if worst_facet is None:
            break
        
        v1_idx, v2_idx, v3_idx = worst_facet
        v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]
        
        # Move toward circumcenter with very small step
        p1_new = v1 + step_size * (worst_center - v1)
        p2_new = v2 + step_size * (worst_center - v2)
        p3_new = v3 + step_size * (worst_center - v3)
        
        n1 = np.linalg.norm(p1_new)
        n2 = np.linalg.norm(p2_new)
        n3 = np.linalg.norm(p3_new)
        
        if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
            pts[v1_idx] = p1_new / n1
            pts[v2_idx] = p2_new / n2
            pts[v3_idx] = p3_new / n3
        
        step_size *= decay
        if step_size < 1e-10:
            break
    
    return best, best_radius

def _smooth_covering_radius(pts, grid_size=5000):
    """Smooth approximation of covering radius using grid sampling."""
    M = grid_size
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    min_dots = dots.max(axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _gradient_step(pts, grid_size=3000, step_size=0.02):
    """Compute gradient of smooth covering radius and take a step."""
    M = grid_size
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute dots: (M, N)
    dots = grid @ pts.T
    
    # For each grid point, the nearest point index
    nearest_idx = np.argmax(dots, axis=1)
    
    # Gradient: d(arccos(y·p_i))/dp_i = -y / sqrt(1 - (y·p_i)^2)
    # But we only care about the max over grid points
    # Use soft-max to smooth
    
    beta = 30.0
    shifted = beta * dots
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)  # (M, N)
    
    # g(y) = weighted sum of dots
    g = np.sum(weights * dots, axis=1)
    g = np.clip(g, -1.0 + 1e-12, 1.0 - 1e-12)
    d = np.arccos(g)
    
    # soft-max over y
    beta2 = 30.0
    shifted2 = beta2 * d
    max_shifted2 = np.max(shifted2)
    exp_shifted2 = np.exp(shifted2 - max_shifted2)
    y_weights = exp_shifted2 / np.sum(exp_shifted2)  # (M,)
    
    # Gradient w.r.t. p_i
    # d/dp_i [arccos(g)] = -dg/dp_i / sqrt(1-g^2)
    # dg/dp_i = sum_y y_weights[y] * weights[y,i] * y[y]
    # So dE/dp_i = -sum_y y_weights[y] * weights[y,i] * grid[y] / sqrt(1-g[y]^2)
    
    sqrt_term = np.sqrt(np.maximum(1.0 - g**2, 1e-12))
    inv_sqrt = 1.0 / sqrt_term
    
    # grad: (N, 3)
    grad = np.zeros((N, 3))
    for i in range(N):
        # weights[:, i] is (M,)
        weighted = y_weights * weights[:, i] * inv_sqrt
        grad[i] = -np.sum(weighted[:, None] * grid, axis=0)
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-12)
    grad = grad / grad_norm
    
    # Take step
    new_pts = pts - step_size * grad
    # Renormalize
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    new_pts = new_pts / norms
    
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best_points)
    
    np.random.seed(42)
    
    # Phase 1: Rapid local refinement from Fibonacci
    best_points, best_radius = _local_refine(points, max_iters=2000, step_size=0.05, decay=0.999, deadline=deadline)
    
    # Phase 2: Multiple restarts with simulated annealing
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate initial points
        if restart == 0:
            pts = best_points.copy()
        elif restart == 1:
            pts = _fibonacci_sphere(N)
            # Slight perturbation
            noise = np.random.randn(N, 3) * 0.05
            pts = pts + noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Random points with some structure
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            # Mix with best solution
            mix = 0.3
            pts = (1 - mix) * pts + mix * best_points
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        current_radius, _, _, _ = _compute_covering_metrics(pts)
        
        # Simulated annealing
        temperature = 0.02
        cooling = 0.995
        min_temperature = 1e-5
        max_sa_iters = 3000
        stuck_counter = 0
        best_sa_radius = current_radius
        best_sa_pts = pts.copy()
        
        for it in range(max_sa_iters):
            if time.time() > deadline - 10:
                break
            
            # Choose a random point to move
            idx = np.random.randint(N)
            
            # Compute move direction: toward worst facet center if this point is in worst facet
            current_radius, worst_facet, worst_center, all_radii = _compute_covering_metrics(pts)
            
            if worst_facet is not None:
                v1_idx, v2_idx, v3_idx = worst_facet
                if idx in [v1_idx, v2_idx, v3_idx]:
                    # Move toward circumcenter
                    target = worst_center
                    move = target - pts[idx]
                    move_norm = np.linalg.norm(move)
                    if move_norm > 1e-10:
                        move = move / move_norm * temperature
                else:
                    # Random small move
                    move = np.random.randn(3) * temperature
            else:
                move = np.random.randn(3) * temperature
            
            new_pt = pts[idx] + move
            new_pt_norm = np.linalg.norm(new_pt)
            if new_pt_norm < 1e-10:
                continue
            new_pt = new_pt / new_pt_norm
            
            old_pt = pts[idx].copy()
            pts[idx] = new_pt
            
            new_radius, _, _, _ = _compute_covering_metrics(pts)
            
            # Accept if better, or with probability exp(-delta/T)
            delta = new_radius - current_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temperature, 1e-10)):
                current_radius = new_radius
                if new_radius < best_sa_radius:
                    best_sa_radius = new_radius
                    best_sa_pts = pts.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                pts[idx] = old_pt
                stuck_counter += 1
            
            temperature *= cooling
            if temperature < min_temperature:
                break
            
            # If stuck for too long, do a gradient step
            if stuck_counter > 200 and it % 50 == 0:
                pts = _gradient_step(pts, grid_size=3000, step_size=0.01)
                current_radius, _, _, _ = _compute_covering_metrics(pts)
                if current_radius < best_sa_radius:
                    best_sa_radius = current_radius
                    best_sa_pts = pts.copy()
                    stuck_counter = 0
        
        # Update global best
        if best_sa_radius < best_radius:
            best_radius = best_sa_radius
            best_points = best_sa_pts.copy()
        
        # Local refine the best from this restart
        refined_pts, refined_radius = _local_refine(best_sa_pts, max_iters=500, step_size=0.02, decay=0.999, deadline=deadline)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
    
    # Phase 3: Final polish with high precision
    if time.time() < deadline - 5:
        polished_pts, polished_radius = _high_precision_polish(best_points.copy(), deadline)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished_pts.copy()
    
    # Ensure final points are normalized and valid
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
