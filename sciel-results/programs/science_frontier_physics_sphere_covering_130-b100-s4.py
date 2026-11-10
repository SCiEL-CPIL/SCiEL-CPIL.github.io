"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    Uses vectorized circumradius computation for triangular facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        
        # Handle triangular facets (most common in 3D convex hull)
        tri_facets = [f for f in facets if len(f) == 3]
        
        if len(tri_facets) > 0:
            tri_facets = np.array(tri_facets)
            a = points[tri_facets[:, 0]]
            b = points[tri_facets[:, 1]]
            c = points[tri_facets[:, 2]]
            
            # Compute cross products
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            
            # Triple product: a . (b x c)
            triple = np.einsum('ij,ij->i', a, np.cross(b, c))
            denom = 4.0 * np.abs(triple)
            
            # Avoid division by zero
            valid = denom > 1e-15
            if np.any(valid):
                norms_ab = np.linalg.norm(ab[valid], axis=1)
                norms_bc = np.linalg.norm(bc[valid], axis=1)
                norms_ca = np.linalg.norm(ca[valid], axis=1)
                
                sin_R = (norms_ab * norms_bc * norms_ca) / denom[valid]
                sin_R = np.clip(sin_R, 0.0, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, np.max(R))
        
        # Handle non-triangular facets (rare, but possible)
        non_tri_facets = [f for f in facets if len(f) != 3]
        for simplex in non_tri_facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            if n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf


def _get_worst_facet(points):
    """
    Returns the index of the facet with the largest circumradius, 
    the circumcenter direction, and the circumradius.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_radius = -1.0
        worst_idx = -1
        worst_center = None
        
        for i, simplex in enumerate(facets):
            pts = points[simplex]
            if len(pts) != 3:
                continue
            a, b, c = pts[0], pts[1], pts[2]
            v1 = a - b
            v2 = a - c
            v = np.cross(v1, v2)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            x = v / norm_v
            cos_r = np.clip(np.dot(x, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_radius:
                max_radius = r
                worst_idx = i
                worst_center = x
        
        if worst_idx >= 0:
            return worst_idx, worst_center, max_radius
        return None, None, 0.0
    except Exception:
        return None, None, 0.0


def _optimize_worst_facet(points, max_time=5.0):
    """
    Try to reduce the worst facet by moving its vertices.
    Returns improved points and radius.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_fast(best_points)
    
    # Get the worst facet
    worst_idx, worst_center, worst_r = _get_worst_facet(best_points)
    if worst_idx is None:
        return best_points, best_radius
    
    # Try to move vertices of the worst facet
    # The facet vertices are in hull.simplices[worst_idx]
    # We need to get the hull to find the vertices
    try:
        hull = ConvexHull(best_points)
        facet = hull.simplices[worst_idx]
    except:
        return best_points, best_radius
    
    # Try moving each vertex of the facet slightly towards the circumcenter
    # or in a direction that reduces the circumradius
    for _ in range(50):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for vi in facet:
            # Try small perturbations
            for scale in [0.001, 0.005, 0.01, 0.02]:
                candidate = best_points.copy()
                # Move vertex vi slightly towards the circumcenter direction
                # or try random small perturbations
                for _ in range(3):
                    perturb = np.random.randn(3) * scale
                    candidate[vi] = best_points[vi] + perturb
                    candidate[vi] /= np.linalg.norm(candidate[vi])
                    r = _covering_radius_fast(candidate)
                    if r < best_radius:
                        best_points = candidate
                        best_radius = r
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best_points, best_radius


def _optimize_from_start(points, max_time=20.0):
    """
    Local optimization using a combination of worst-facet targeting 
    and general hill-climbing.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_fast(best_points)
    
    # Phase 1: Target the worst facet
    while time.time() - start_time < max_time * 0.5:
        opt_points, opt_radius = _optimize_worst_facet(best_points, max_time=2.0)
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
        else:
            break
    
    # Phase 2: General hill-climbing with smaller steps
    step_size = 0.005
    while time.time() - start_time < max_time:
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            # Random small perturbation
            perturb = np.random.randn(3) * step_size
            candidate = best_points.copy()
            candidate[i] = best_points[i] + perturb
            candidate[i] /= np.linalg.norm(candidate[i])
            r = _covering_radius_fast(candidate)
            if r < best_radius:
                best_points = candidate
                best_radius = r
                improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points
    best_radius = _covering_radius_fast(best_points)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random start
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        
        # Optimize from this start
        opt_points, opt_radius = _optimize_from_start(rand_points, max_time=15.0)
        
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
    
    # Final optimization from the best point found
    if time.time() < deadline - 5.0:
        final_points, final_radius = _optimize_from_start(best_points, max_time=25.0)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
