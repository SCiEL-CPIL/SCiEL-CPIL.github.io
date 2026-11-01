import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _true_radius_deg(config):
    """
    Compute the true covering radius in degrees by finding the circumcenters
    of the convex hull facets and checking their angular distance to the
    nearest configuration point.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback: use dense sampling
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    # For each facet, find the circumcenter of the triangle on the unit sphere
    max_angle = 0.0
    for simplex in hull.simplices:
        # Get the three points of this facet
        pts = config[simplex]  # (3, 3)
        
        # Compute circumcenter of the triangle formed by these 3 points
        # The circumcenter lies on the plane through the 3 points
        # and is equidistant from all three.
        
        # Method: solve for point c such that |c - p_i|^2 = |c - p_j|^2 for all i,j
        # This gives: 2 * c . (p_j - p_i) = |p_j|^2 - |p_i|^2 = 0 (since all on unit sphere)
        # So c is perpendicular to (p_j - p_i) and (p_k - p_i)
        # i.e., c is parallel to the cross product (p_j - p_i) x (p_k - p_i)
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        
        if c_norm < 1e-12:
            # Degenerate facet, skip
            continue
        
        c = c / c_norm
        
        # Now we have the circumcenter direction. But we need to verify it's
        # actually the circumcenter on the sphere (equidistant from all 3 points).
        # The circumcenter of a spherical triangle is the point on the sphere
        # equidistant from all three vertices.
        
        # Check: compute angular distances from c to each vertex
        dots = np.clip(pts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # The circumcenter should have equal angular distances to all 3 vertices
        # If not, the facet's covering point might be a vertex instead
        if np.max(angles) - np.min(angles) > 1e-6:
            # Not a proper circumcenter; the worst point in this facet is a vertex
            # The maximum angular distance in this facet is the max distance from
            # any point in the facet to the nearest config point.
            # For a spherical triangle, the worst point is either a circumcenter
            # (if it lies within the triangle) or a vertex.
            # Since the circumcenter direction doesn't give equal distances,
            # the circumcenter is outside the triangle, and the worst point is
            # a vertex.
            max_in_facet = np.max(angles)
        else:
            # Circumcenter is valid and lies within the triangle
            max_in_facet = angles[0]  # All three are equal
        
        # Now check: is this circumcenter's distance to the NEAREST config point
        # actually the covering radius contribution? We need to check that no other
        # config point is closer to this circumcenter than the facet's vertices.
        
        dots_all = np.clip(config @ c, -1.0, 1.0)
        min_angle_to_c = np.arccos(np.max(dots_all))
        
        # The covering radius contribution from this facet is the angular distance
        # from the circumcenter to the nearest config point
        if min_angle_to_c > max_angle:
            max_angle = min_angle_to_c
    
    return np.degrees(max_angle)

def _circumcenters(config):
    """
    Return the circumcenters of all convex hull facets that are valid
    (lie within the facet) along with their covering radius contribution.
    Returns: list of (circumcenter, angular_distance_to_nearest)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return []
    
    results = []
    for simplex in hull.simplices:
        pts = config[simplex]
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        
        if c_norm < 1e-12:
            continue
        
        c = c / c_norm
        
        # Check if circumcenter is valid (equidistant from all 3 vertices)
        dots = np.clip(pts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        
        if np.max(angles) - np.min(angles) > 1e-6:
            continue
        
        # Check that no other config point is closer
        dots_all = np.clip(config @ c, -1.0, 1.0)
        nearest_idx = np.argmax(dots_all)
        min_angle = np.arccos(dots_all[nearest_idx])
        
        # Only include if this is actually the farthest point from config
        # (i.e., the circumcenter is the worst point in its facet)
        results.append((c, min_angle))
    
    return results

def _optimize(config, deadline):
    """
    Optimize the configuration using gradient-based methods on the
    critical points (circumcenters of hull facets).
    """
    start_time = time.time()
    config = config.copy()
    
    # First, run Lloyd's algorithm to get a good initial configuration
    M = 5000
    grid = _fib_sample(M)
    
    def lloyd_step(cfg):
        dots = grid @ cfg.T
        max_idx = np.argmax(dots, axis=1)
        
        new_cfg = np.zeros_like(cfg)
        for i in range(N):
            mask = (max_idx == i)
            count = np.sum(mask)
            if count > 0:
                center = np.mean(grid[mask], axis=0)
                new_cfg[i] = _normalize(center)
            else:
                new_cfg[i] = np.random.randn(3)
                new_cfg[i] = _normalize(new_cfg[i])
        return new_cfg
    
    # Run Lloyd's algorithm
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        new_config = lloyd_step(config)
        if np.allclose(config, new_config, atol=1e-7):
            break
        config = new_config
    
    # Now do local refinement using critical points
    # Use a surrogate: evaluate covering radius via critical points
    
    def covering_radius_from_criticals(cfg):
        crits = _circumcenters(cfg)
        if len(crits) == 0:
            # Fallback
            return _true_radius_deg(cfg)
        max_ang = max(c[1] for c in crits)
        return np.degrees(max_ang)
    
    def objective(params):
        # params: flattened 3N vector of points on sphere
        pts = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        pts = pts / norms
        
        # Check validity
        if not np.all(np.isfinite(pts)):
            return 1e6
        
        # Check if in closed hemisphere
        # Compute the maximum dot product with any direction
        # If all points are in a hemisphere, there exists a direction d such that
        # d . p_i >= 0 for all i
        # Simple check: compute the convex hull and see if origin is inside
        try:
            hull = ConvexHull(pts)
            # Check if origin is in the convex hull
            # The origin is in the convex hull iff for every facet,
            # the origin is on the same side as the interior
            # Simpler: check if there's a vector d with d . p_i < 0 for all i
            # This is a linear programming problem, but let's use a heuristic
            # Compute the mean and check
            pass
        except Exception:
            pass
        
        return covering_radius_from_criticals(pts)
    
    # Gradient-free local optimization using Nelder-Mead or similar
    # But that's slow for 150 dimensions. Instead, use a custom local search.
    
    current_score = covering_radius_from_criticals(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.05
    stagnation = 0
    
    while time.time() < deadline - 5:
        # Find the worst critical point
        crits = _circumcenters(config)
        if len(crits) == 0:
            break
        
        # Find the critical point with maximum angular distance
        worst_idx = np.argmax([c[1] for c in crits])
        worst_point = crits[worst_idx][0]
        worst_angle = crits[worst_idx][1]
        
        # Find which config points are the vertices of the facet containing this
        # Actually, we need to find the nearest config points to the worst point
        dots = np.clip(worst_point @ config.T, -1.0, 1.0)
        nearest_idx = np.argmax(dots)
        
        # Move the nearest config point towards the worst point
        v = worst_point - config[nearest_idx]
        v_proj = v - np.dot(v, config[nearest_idx]) * config[nearest_idx]
        v_norm = np.linalg.norm(v_proj)
        
        if v_norm < 1e-10:
            # Try moving a different point
            # Find the second nearest
            sorted_idx = np.argsort(-dots)
            moved = False
            for idx in sorted_idx:
                if idx == nearest_idx:
                    continue
                v2 = worst_point - config[idx]
                v2_proj = v2 - np.dot(v2, config[idx]) * config[idx]
                v2_norm = np.linalg.norm(v2_proj)
                if v2_norm > 1e-10:
                    new_pt = config[idx] + step_size * v2_proj
                    new_pt = _normalize(new_pt)
                    old_pt = config[idx].copy()
                    config[idx] = new_pt
                    
                    new_score = covering_radius_from_criticals(config)
                    if new_score < current_score:
                        current_score = new_score
                        stagnation = 0
                        if new_score < best_score:
                            best_score = new_score
                            best_config = config.copy()
                    else:
                        config[idx] = old_pt
                    moved = True
                    break
            
            if not moved:
                break
            continue
        
        new_pt = config[nearest_idx] + step_size * v_proj
        new_pt = _normalize(new_pt)
        old_pt = config[nearest_idx].copy()
        config[nearest_idx] = new_pt
        
        new_score = covering_radius_from_criticals(config)
        if new_score < current_score:
            current_score = new_score
            stagnation = 0
            if new_score < best_score:
                best_score = new_score
                best_config = config.copy()
        else:
            config[nearest_idx] = old_pt
            stagnation += 1
            step_size *= 0.9
            if step_size < 0.001 or stagnation > 20:
                break
    
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 seconds safety margin
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run optimization
    config = _optimize(config, deadline)
    
    # Final normalization and validation
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                # Perturb
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
    
    return config
