import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130
DEADLINE = 135.0  # 15s safety margin before 150s kill

def compute_facet_data(points):
    """Compute circumradius and circumcenter for each facet."""
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[~valid] = [0, 0, 1]
    
    if np.any(valid):
        a_v = a[valid]
        b_v = b[valid]
        c_v = c[valid]
        n_v = n[valid]
        norms_v = norms[valid]
        
        u = n_v / norms_v[:, np.newaxis]
        centroids = (a_v + b_v + c_v) / 3.0
        dots = np.sum(u * centroids, axis=1)
        u[dots < 0] *= -1
        
        cos_theta = np.sum(u * a_v, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[valid] = np.arccos(cos_theta)
        centers[valid] = u
    
    return radii, centers, simplices

def get_covering_radius(points):
    """Get the covering radius (max facet circumradius) in radians."""
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def greedy_init(n_points, seed=42):
    """Farthest Point Sampling initialization with more candidates for better quality."""
    np.random.seed(seed)
    # Generate a large pool of candidates
    pool_size = 50000
    candidates = np.random.randn(pool_size, 3)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    
    points = [candidates[0].copy()]
    used = np.zeros(pool_size, dtype=bool)
    used[0] = True
    
    for _ in range(n_points - 1):
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        max_dist[used] = -np.inf
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx].copy())
        used[best_idx] = True
        
    return np.array(points)

def icosahedral_init(n_points, seed=42):
    """Generate points with icosahedral symmetry as starting structure."""
    np.random.seed(seed)
    
    # Start with icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add more points by subdividing faces
    points = list(ico)
    
    while len(points) < n_points:
        # Find the largest facet and add a point near its center
        pts = np.array(points)
        radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            break
        max_idx = np.argmax(radii)
        center = centers[max_idx]
        # Add point slightly perturbed from center
        new_pt = center + np.random.randn(3) * 0.01
        new_pt /= np.linalg.norm(new_pt)
        points.append(new_pt)
    
    return np.array(points[:n_points])

def finite_diff_gradient(points, step=1e-5):
    """Compute gradient of covering radius using finite differences in tangent space."""
    n = len(points)
    grad = np.zeros_like(points)
    base_radius = get_covering_radius(points)
    
    for i in range(n):
        # Compute two tangent directions at point i
        p = points[i]
        # Find two vectors orthogonal to p
        v1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(p, v1)) > 0.9:
            v1 = np.array([0.0, 1.0, 0.0])
        t1 = v1 - np.dot(v1, p) * p
        t1 /= np.linalg.norm(t1)
        
        t2 = np.cross(p, t1)
        t2 /= np.linalg.norm(t2)
        
        for t_dir in [t1, t2]:
            # Forward difference
            p_plus = p + step * t_dir
            p_plus /= np.linalg.norm(p_plus)
            pts_plus = points.copy()
            pts_plus[i] = p_plus
            r_plus = get_covering_radius(pts_plus)
            
            # Backward difference
            p_minus = p - step * t_dir
            p_minus /= np.linalg.norm(p_minus)
            pts_minus = points.copy()
            pts_minus[i] = p_minus
            r_minus = get_covering_radius(pts_minus)
            
            # Central difference gradient component
            grad[i] += (r_plus - r_minus) / (2 * step) * t_dir
    
    return grad

def polish_gradient(points, time_budget, start_alpha=0.05):
    """High-precision gradient polish using finite differences."""
    start = time.time()
    best_local = points.copy()
    best_local_radius = get_covering_radius(points)
    alpha = start_alpha
    
    # Use fewer but more careful steps since finite diff is expensive
    max_steps = 200
    
    for step in range(max_steps):
        if time.time() - start > time_budget:
            break
        
        # Compute gradient (expensive but reliable)
        grad = finite_diff_gradient(points, step=1e-5)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Take step with line search
        best_r = best_local_radius
        best_pt = points.copy()
        
        for trial_alpha in [alpha, alpha * 0.5, alpha * 0.25, alpha * 0.1]:
            new_points = points - trial_alpha * grad_tangent
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            new_r = get_covering_radius(new_points)
            if new_r < best_r:
                best_r = new_r
                best_pt = new_points.copy()
        
        points = best_pt
        
        if best_r < best_local_radius:
            best_local_radius = best_r
            best_local = points.copy()
        
        # Decay step size
        alpha *= 0.9
        if alpha < 1e-5:
            break
    
    return best_local

def polish_facet_push(points, time_budget):
    """Final polish by pushing vertices of largest facets."""
    start = time.time()
    best_local = points.copy()
    best_local_radius = get_covering_radius(points)
    T = 0.03
    
    for step in range(2000):
        if time.time() - start > time_budget:
            break
        
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
        
        # Find facets above threshold
        threshold = np.percentile(radii, 90)
        mask = radii > threshold
        
        if not np.any(mask):
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
        
        # Push vertices toward circumcenters with proper spherical geometry
        updates = np.zeros_like(points)
        for m in np.where(mask)[0]:
            idx = simplices[m]
            i_idx, j_idx, k_idx = idx
            center = centers[m]
            r = radii[m]
            
            # Move each vertex toward the circumcenter on the sphere
            step_size = min(T, 0.1) * r
            
            for idx_i in [i_idx, j_idx, k_idx]:
                p = points[idx_i]
                c = center
                # Spherical interpolation direction
                dot_pc = np.dot(p, c)
                dot_pc = np.clip(dot_pc, -1.0, 1.0)
                angle = np.arccos(dot_pc)
                if angle < 1e-10:
                    continue
                # Move along great circle toward center
                s1 = np.sin(angle)
                new_p = (np.sin((1-step_size)*angle)/s1) * p + (np.sin(step_size*angle)/s1) * c
                new_p /= np.linalg.norm(new_p)
                updates[idx_i] += new_p - p
        
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        T *= 0.995
        if T < 1e-6:
            break
    
    return best_local

def construct():
    start_time = time.time()
    
    best_points = None
    best_radius = np.pi
    
    # Generate diverse initializations (fewer but higher quality)
    initializations = []
    
    # 1. Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(("fib", init_fib))
    
    # 2. Greedy (FPS) with good seed
    init_greedy = greedy_init(N, seed=10)
    initializations.append(("greedy10", init_greedy))
    
    # 3. Icosahedral-based
    init_ico = icosahedral_init(N, seed=42)
    initializations.append(("ico42", init_ico))
    
    # 4. Fibonacci + small noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.08
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(("fib_noise", init_fib_noise))
    
    # Optimization for each initialization
    for init_name, init_points in initializations:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20.0:
            break
        
        points = init_points.copy()
        local_best = points.copy()
        local_best_r = get_covering_radius(points)
        
        # Phase 1: Facet push optimization (main workhorse)
        T = 0.05
        no_improve = 0
        
        for step in range(1500):
            if time.time() - start_time > DEADLINE - 15.0:
                break
            
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) == 0:
                break
            
            max_r = np.max(radii)
            if max_r < local_best_r:
                local_best_r = max_r
                local_best = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 80:
                    break
            
            # Push vertices of large facets
            threshold = np.percentile(radii, 85)
            mask = radii > threshold
            
            if not np.any(mask):
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True
            
            updates = np.zeros_like(points)
            for m in np.where(mask)[0]:
                idx = simplices[m]
                i_idx, j_idx, k_idx = idx
                center = centers[m]
                r = radii[m]
                
                step_size = min(T, 0.08) * r
                updates[i_idx] += step_size * (center - points[i_idx])
                updates[j_idx] += step_size * (center - points[j_idx])
                updates[k_idx] += step_size * (center - points[k_idx])
            
            points = points + updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            T *= 0.997
        
        # Phase 2: Gradient polish (if time allows)
        remaining_for_grad = DEADLINE - 10.0 - (time.time() - start_time)
        if remaining_for_grad > 3.0:
            polished = polish_gradient(local_best, time_budget=min(5.0, remaining_for_grad))
            polished_r = get_covering_radius(polished)
            if polished_r < local_best_r:
                local_best_r = polished_r
                local_best = polished
        
        # Phase 3: Facet push polish
        remaining_for_push = DEADLINE - 5.0 - (time.time() - start_time)
        if remaining_for_push > 2.0:
            polished = polish_facet_push(local_best, time_budget=min(4.0, remaining_for_push))
            polished_r = get_covering_radius(polished)
            if polished_r < local_best_r:
                local_best_r = polished_r
                local_best = polished
        
        # Update global best
        if local_best_r < best_radius:
            best_radius = local_best_r
            best_points = local_best.copy()
        
        print(f"Init {init_name}: radius = {np.degrees(local_best_r):.6f} deg")
    
    # Final polish on the best result
    if best_points is not None:
        remaining = DEADLINE - 3.0 - (time.time() - start_time)
        if remaining > 2.0:
            polished = polish_facet_push(best_points, time_budget=min(5.0, remaining))
            polished_r = get_covering_radius(polished)
            if polished_r < best_radius:
                best_radius = polished_r
                best_points = polished
    
    if best_points is None:
        best_points = initializations[0][1]
    
    # Final verification
    final_r = get_covering_radius(best_points)
    print(f"Final radius: {np.degrees(final_r):.6f} deg")
    
    return best_points
