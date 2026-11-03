import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    edges = [
        (p2 - p1, p3 - p1),
        (p1 - p2, p3 - p2),
        (p1 - p3, p2 - p3)
    ]
    
    for v1, v2 in edges:
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        n = n / n_norm
        center = n.copy()
        
        dot1 = np.dot(center, p1)
        dot2 = np.dot(center, p2)
        dot3 = np.dot(center, p3)
        
        if dot1 > 0 and dot2 > 0 and dot3 > 0:
            return center
        else:
            return -center
    
    # Fallback: average of the three points, normalized
    center = (p1 + p2 + p3) / 3.0
    norm = np.linalg.norm(center)
    if norm < 1e-10:
        return p1.copy()
    return center / norm

def _compute_circumradii(pts, hull):
    """
    Compute angular circumradii for all triangular facets of the convex hull.
    Returns array of circumradii (in radians) and the circumcenters.
    """
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for k, tri in enumerate(simplices):
        v0 = pts[tri[0]]
        v1 = pts[tri[1]]
        v2 = pts[tri[2]]
        
        c = _circumcenter_robust(v0, v1, v2)
        centers[k] = c
        
        dot = np.clip(np.dot(c, v0), -1.0, 1.0)
        radii[k] = np.arccos(dot)
        
    return radii, centers

def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, None, None
        
    radii, centers = _compute_circumradii(pts, hull)
    max_radius = np.max(radii)
    return np.degrees(max_radius), radii, centers

def _soft_max(radii, beta=50.0):
    """
    Compute soft-max of circumradii using log-sum-exp for smooth optimization.
    As beta -> inf, this approaches the true max.
    """
    # Shift for numerical stability
    r_max = np.max(radii)
    shifted = radii - r_max
    # Soft max: r_max + (1/beta) * log(sum(exp(beta * r_i / beta)))
    # = r_max + (1/beta) * log(sum(exp(r_i - r_max)))
    log_sum = np.log(np.sum(np.exp(shifted)))
    return r_max + log_sum / beta

def _soft_max_gradient(radii, beta=50.0):
    """
    Compute gradient of soft-max with respect to radii.
    Returns weights that sum to 1.
    """
    r_max = np.max(radii)
    shifted = radii - r_max
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    return weights

def _repulsive_force(pts):
    """Compute repulsive force between all pairs of points to prevent clustering."""
    n = len(pts)
    rep_force = np.zeros_like(pts)
    
    # Compute pairwise distances
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                d_ij = D[i, j]
                if d_ij < 1e-6:
                    f = 100.0
                else:
                    f = 1.0 / (d_ij * d_ij)
                rep_force[i] += f * (pts[i] - pts[j])
                
    return rep_force

def _compute_soft_max_gradient(pts, hull, radii, centers, beta=50.0):
    """
    Compute gradient of soft-max circumradius with respect to point positions.
    
    For each facet k with circumradius r_k and circumcenter c_k, the derivative of r_k
    with respect to a point v_i is related to the projection of (c_k - v_i) onto the
    tangent plane at v_i.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    weights = _soft_max_gradient(radii, beta)
    
    for k, tri in enumerate(hull.simplices):
        w = weights[k]
        if w < 1e-10:
            continue
            
        c = centers[k]
        r = radii[k]
        
        # Derivative of circumradius w.r.t. each vertex
        # The circumradius r satisfies: for vertex v, the angle at v subtended by the
        # arc opposite to v is related to r. 
        # A simpler approach: the gradient of the distance from c to v_i on the sphere
        # projected onto the tangent plane.
        
        for idx in tri:
            v = pts[idx]
            # Vector from v to c projected onto tangent plane at v
            proj = np.dot(v, c) * c
            tangent = c - proj
            norm_t = np.linalg.norm(tangent)
            
            if norm_t > 1e-10:
                # The rate of change of the angle arccos(dot(c,v)) w.r.t. v moving
                # in direction d is -dot(c, d) / sqrt(1 - dot(c,v)^2)
                # dot(c,v) = cos(r), so sqrt(1 - cos^2(r)) = sin(r)
                dot_cv = np.dot(c, v)
                sin_r = np.sqrt(np.maximum(0, 1.0 - dot_cv * dot_cv))
                
                if sin_r > 1e-10:
                    # Gradient contribution: -w * (c - proj) / sin_r
                    grad[idx] -= w * tangent / sin_r
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_init()
    
    best_cr = 180.0
    best_pts = pts.copy()
    
    # Pre-compute initial hull
    try:
        hull = ConvexHull(pts)
        radii, centers = _compute_circumradii(pts, hull)
        current_cr = np.degrees(np.max(radii))
    except Exception:
        return pts
        
    if current_cr < best_cr:
        best_cr = current_cr
        best_pts = pts.copy()
        
    step_size_greedy = 0.05
    step_size_lloyd = 0.02
    step_size_grad = 0.01
    no_improve_count = 0
    max_no_improve = 30
    iteration = 0
    
    # Phase 1: Use greedy/Lloyd for initial optimization
    phase1_deadline = start_time + 40.0
    
    while time.time() < phase1_deadline:
        iteration += 1
        
        # Alternate between Lloyd and Greedy
        use_lloyd = (iteration % 6 == 0)
        
        if use_lloyd:
            # Lloyd step: move each point towards centroid of adjacent facet circumcenters
            simplices = hull.simplices
            point_to_facets = [[] for _ in range(N)]
            for k, tri in enumerate(simplices):
                for idx in tri:
                    point_to_facets[idx].append(k)
                    
            new_pts = pts.copy()
            for i in range(N):
                facets = point_to_facets[i]
                if not facets:
                    continue
                    
                sum_c = np.sum(centers[facets], axis=0)
                norm_c = np.linalg.norm(sum_c)
                if norm_c < 1e-10:
                    continue
                    
                target = sum_c / norm_c
                new_pts[i] = pts[i] + step_size_lloyd * (target - pts[i])
                
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            pts = new_pts
        else:
            # Greedy: Shrink the largest facet
            max_idx = np.argmax(radii)
            max_rad = radii[max_idx]
            
            if max_rad < 1e-6:
                break
                
            tri = hull.simplices[max_idx]
            c = centers[max_idx]
            
            moves = np.zeros(3)
            for idx in tri:
                v = pts[idx]
                # Project c onto tangent plane at v
                proj = np.dot(v, c) * c
                tangent = c - proj
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    moves += tangent / norm_t
                    
            move_norm = np.linalg.norm(moves)
            if move_norm < 1e-10:
                # If stuck, try a small random perturbation
                pts += np.random.normal(0, 0.001, (N, 3))
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            else:
                moves = moves / move_norm
                for idx in tri:
                    pts[idx] += step_size_greedy * moves
                    pts[idx] /= np.linalg.norm(pts[idx])
        
        # Update hull and radii
        try:
            hull = ConvexHull(pts)
            radii, centers = _compute_circumradii(pts, hull)
            new_cr = np.degrees(np.max(radii))
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Apply repulsive force to break symmetry
                    rep = _repulsive_force(pts)
                    pts = pts + 0.01 * rep
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                    
                    # Reduce step sizes
                    step_size_greedy *= 0.9
                    step_size_lloyd *= 0.9
                    no_improve_count = 0
                    if step_size_greedy < 1e-5:
                        break
        except Exception:
            # Hull failed, revert to best
            pts = best_pts.copy()
            break
    
    # Phase 2: Use smooth gradient-based optimization with soft-max
    # This helps escape local minima by considering all facets, not just the max
    phase2_deadline = deadline - 5.0  # Leave 5s for final polish
    
    # Start from best found
    pts = best_pts.copy()
    
    try:
        hull = ConvexHull(pts)
        radii, centers = _compute_circumradii(pts, hull)
    except Exception:
        return best_pts
    
    # Anneal beta from low to high for smooth transition
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0]
    grad_step = 0.005
    
    for beta in beta_schedule:
        if time.time() > phase2_deadline:
            break
            
        # Multiple iterations at each beta
        for inner_iter in range(20):
            if time.time() > phase2_deadline:
                break
                
            try:
                hull = ConvexHull(pts)
                radii, centers = _compute_circumradii(pts, hull)
            except Exception:
                break
                
            # Compute gradient of soft-max
            grad = _compute_soft_max_gradient(pts, hull, radii, centers, beta)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
                
            grad = grad / grad_norm
            
            # Take gradient step
            pts_new = pts - grad_step * grad
            pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
            
            # Check if this improves the actual max circumradius
            try:
                hull_new = ConvexHull(pts_new)
                radii_new, centers_new = _compute_circumradii(pts_new, hull_new)
                new_cr = np.degrees(np.max(radii_new))
                
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_pts = pts_new.copy()
                    pts = pts_new
                    hull = hull_new
                    radii = radii_new
                    centers = centers_new
                else:
                    # Revert if not improved
                    break
            except Exception:
                break
    
    # Phase 3: Final polish with smaller steps on the best solution
    if time.time() < deadline - 2.0:
        pts = best_pts.copy()
        try:
            hull = ConvexHull(pts)
            radii, centers = _compute_circumradii(pts, hull)
        except Exception:
            return best_pts
        
        polish_step = 0.002
        for polish_iter in range(50):
            if time.time() > deadline - 2.0:
                break
                
            # Find largest facet
            max_idx = np.argmax(radii)
            max_rad = radii[max_idx]
            
            if max_rad < 1e-7:
                break
                
            tri = hull.simplices[max_idx]
            c = centers[max_idx]
            
            moves = np.zeros(3)
            for idx in tri:
                v = pts[idx]
                proj = np.dot(v, c) * c
                tangent = c - proj
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    moves += tangent / norm_t
                    
            move_norm = np.linalg.norm(moves)
            if move_norm < 1e-10:
                break
                
            moves = moves / move_norm
            for idx in tri:
                pts[idx] += polish_step * moves
                pts[idx] /= np.linalg.norm(pts[idx])
            
            try:
                hull = ConvexHull(pts)
                radii, centers = _compute_circumradii(pts, hull)
                new_cr = np.degrees(np.max(radii))
                
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_pts = pts.copy()
            except Exception:
                break
    
    return best_pts

construct()
