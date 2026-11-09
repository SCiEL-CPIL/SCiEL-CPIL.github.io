import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import itertools
from scipy.optimize import minimize

_DEADLINE = time.time() + 585.0  # Safety margin
_COMBOS = np.array(list(itertools.combinations(range(13), 3)))

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = _COMBOS
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _ratio(points):
    """Compute min_triangle_area / hull_area. Returns 0 if invalid."""
    if not np.all(np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    if ma < 1e-12:
        return 0.0
    return ma / ha

def _identify_hull_indices(points):
    """Return sorted list of indices on the convex hull."""
    try:
        hull = ConvexHull(points)
        return sorted(hull.vertices)
    except Exception:
        return list(range(len(points)))

def _lloyd_relax_interior(points, hull_indices, n_iters=50):
    """
    Lloyd's algorithm: relax interior points to Voronoi centroids.
    Hull points are fixed. This balances local density.
    """
    n = len(points)
    interior_indices = [i for i in range(n) if i not in hull_indices]
    if not interior_indices:
        return points.copy()
    
    points = points.copy()
    
    for it in range(n_iters):
        if time.time() > _DEADLINE - 30:
            break
        try:
            delaunay = Delaunay(points)
        except Exception:
            break
        
        # For each interior point, compute its Voronoi cell centroid
        # Using the dual of Delaunay: neighbors of point i define its Voronoi cell
        for idx in interior_indices:
            # Find neighbors (shared edges) of point idx in Delaunay triangulation
            neighbors = set()
            for simplex in delaunay.simplices:
                if idx in simplex:
                    for j in simplex:
                        if j != idx:
                            neighbors.add(j)
            
            if not neighbors:
                continue
            
            # Compute weighted centroid using circumcenters of adjacent triangles
            # Simpler: use average of neighbors as a proxy for Voronoi centroid
            neighbor_points = points[list(neighbors)]
            # Weight by inverse distance for better balance
            dists = np.linalg.norm(neighbor_points - points[idx], axis=1)
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / (dists ** 2)
            weights /= weights.sum()
            
            new_pos = np.average(neighbor_points, axis=0, weights=weights)
            
            # Keep interior point inside hull: project if needed
            # Simple check: ensure it's within bounding box of hull
            hull_pts = points[hull_indices]
            if new_pos[0] < hull_pts[:, 0].min() + 0.01 or new_pos[0] > hull_pts[:, 0].max() - 0.01:
                new_pos[0] = np.clip(new_pos[0], hull_pts[:, 0].min() + 0.01, hull_pts[:, 0].max() - 0.01)
            if new_pos[1] < hull_pts[:, 1].min() + 0.01 or new_pos[1] > hull_pts[:, 1].max() - 0.01:
                new_pos[1] = np.clip(new_pos[1], hull_pts[:, 1].min() + 0.01, hull_pts[:, 1].max() - 0.01)
            
            points[idx] = new_pos
    
    return points

def _lloyd_with_voronoi(points, hull_indices, n_iters=30):
    """
    More accurate Lloyd relaxation using actual Voronoi cell centroids.
    """
    n = len(points)
    interior_indices = [i for i in range(n) if i not in hull_indices]
    if not interior_indices:
        return points.copy()
    
    points = points.copy()
    
    for it in range(n_iters):
        if time.time() > _DEADLINE - 30:
            break
        try:
            delaunay = Delaunay(points)
        except Exception:
            break
        
        for idx in interior_indices:
            # Get all triangles containing this vertex
            tri_indices = []
            for ti, simplex in enumerate(delaunay.simplices):
                if idx in simplex:
                    tri_indices.append(ti)
            
            if not tri_indices:
                continue
            
            # Compute Voronoi cell vertices: circumcenters of adjacent triangles
            # Plus the midpoints of edges to neighbors (for boundary of cell)
            # Approximate: average of circumcenters of adjacent triangles
            circumcenters = []
            for ti in tri_indices:
                tri = delaunay.simplices[ti]
                p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
                # Circumcenter formula
                ax, ay = p1
                bx, by = p2
                cx, cy = p3
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                circumcenters.append([ux, uy])
            
            if circumcenters:
                circumcenters = np.array(circumcenters)
                # Centroid of circumcenters approximates Voronoi centroid
                new_pos = circumcenters.mean(axis=0)
                
                # Clip to stay inside
                hull_pts = points[hull_indices]
                new_pos[0] = np.clip(new_pos[0], hull_pts[:, 0].min() + 0.02, hull_pts[:, 0].max() - 0.02)
                new_pos[1] = np.clip(new_pos[1], hull_pts[:, 1].min() + 0.02, hull_pts[:, 1].max() - 0.02)
                
                # Damped update
                points[idx] = 0.5 * points[idx] + 0.5 * new_pos
    
    return points

def _soft_min_ratio(params, temperature):
    """Smooth surrogate: soft-min of triangle areas / hull area."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    if not np.all(np.isfinite(points)):
        return 1e10
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10
    
    idx = _COMBOS
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_a - temperature * log_sum
    
    if soft_min < 1e-12:
        return 1e10
    
    return -(soft_min / ha)

def _exact_neg_ratio(params):
    """Exact objective for final refinement."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    r = _ratio(points)
    if r < 1e-12:
        return 1e10
    return -r

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _concentric_rings(n_inner, n_outer, r_inner=0.3, r_outer=1.0, angle_offset=0.0):
    inner_angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    outer = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    return np.vstack([inner, outer])

def _generate_structured_configs(rng):
    configs = []
    configs.append(_regular_polygon(13, radius=1.0))
    
    for n_in, n_out, r_in, offset in [
        (6, 7, 0.35, 0.0),
        (6, 7, 0.40, np.pi/7),
        (5, 8, 0.35, 0.0),
        (5, 8, 0.40, np.pi/8),
        (4, 9, 0.30, 0.0),
        (4, 9, 0.35, np.pi/9),
        (7, 6, 0.30, 0.0),
        (7, 6, 0.35, np.pi/6),
        (3, 10, 0.25, 0.0),
        (3, 10, 0.30, np.pi/10),
    ]:
        configs.append(_concentric_rings(n_in, n_out, r_in, 1.0, offset))
    
    for _ in range(5):
        base = _concentric_rings(6, 7, 0.38, 1.0, np.pi/7)
        perturbed = base + rng.normal(0, 0.05, (13, 2))
        configs.append(perturbed)
    
    return configs

def _optimize_with_lloyd(points, hull_indices, n_lloyd_iters=30, n_opt_iters=100):
    """
    Hybrid optimization: Lloyd relax interior, then optimize all points
    with soft-min surrogate, alternating.
    """
    if time.time() > _DEADLINE - 60:
        return points
    
    current = points.copy()
    
    for phase in range(3):
        if time.time() > _DEADLINE - 45:
            break
        
        # Step 1: Lloyd relaxation of interior points
        current = _lloyd_with_voronoi(current, hull_indices, n_lloyd_iters=n_lloyd_iters)
        
        # Step 2: Optimize all points with soft-min
        params = current.flatten()
        try:
            res = minimize(lambda p: _soft_min_ratio(p, temperature=0.01), params,
                          method='L-BFGS-B',
                          options={'maxiter': n_opt_iters, 'ftol': 1e-12, 'gtol': 1e-12})
            if np.all(np.isfinite(res.x)):
                current = res.x.reshape((13, 2))
        except Exception:
            pass
        
        # Re-identify hull in case topology changed
        hull_indices = _identify_hull_indices(current)
    
    return current

def _polish_points(points, maxiter=200):
    """High-precision coordinate polish using Powell on exact objective."""
    if time.time() > _DEADLINE - 30:
        return points
    
    params = points.flatten()
    try:
        res = minimize(_exact_neg_ratio, params,
                      method='Powell',
                      options={'maxiter': maxiter, 'xtol': 1e-12, 'ftol': 1e-12})
        opt_points = res.x.reshape((13, 2))
        if np.all(np.isfinite(opt_points)):
            return opt_points
    except Exception:
        pass
    return points

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Generate structured configs and evaluate
    configs = _generate_structured_configs(rng)
    
    for config in configs:
        if time.time() > _DEADLINE - 120:
            break
        
        init_score = _ratio(config)
        if init_score > best_score:
            best_score = init_score
            best_points = config.copy()
    
    # Phase 2: Lloyd + optimization hybrid from best config
    if best_points is not None:
        hull_idx = _identify_hull_indices(best_points)
        
        # Multi-round Lloyd+optimize
        for round_num in range(4):
            if time.time() > _DEADLINE - 90:
                break
            
            hull_idx = _identify_hull_indices(best_points)
            candidate = _optimize_with_lloyd(best_points, hull_idx, 
                                            n_lloyd_iters=20, n_opt_iters=80)
            
            cand_score = _ratio(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        
        # Final polish
        opt_points = _polish_points(best_points, maxiter=150)
        opt_score = _ratio(opt_points)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # Phase 3: Try Lloyd+optimize from other structured configs
    for config in configs[:8]:
        if time.time() > _DEADLINE - 45:
            break
        
        hull_idx = _identify_hull_indices(config)
        candidate = _optimize_with_lloyd(config, hull_idx,
                                        n_lloyd_iters=15, n_opt_iters=60)
        
        cand_score = _ratio(candidate)
        
        # Quick polish
        if cand_score > best_score * 0.95:
            polished = _polish_points(candidate, maxiter=50)
            pol_score = _ratio(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
    
    # Fallback
    if best_points is None or best_score < 0.001:
        best_points = _regular_polygon(13, radius=1.0)
        best_score = _ratio(best_points)
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=1.0)
        
    return best_points
