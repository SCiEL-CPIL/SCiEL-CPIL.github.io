"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _get_worst_facet(points):
    """
    Returns the index of the facet with the largest circumradius and its circumcenter.
    If no valid hull, returns None, None.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = -1.0
        worst_idx = -1
        worst_center = None
        
        for i, simplex in enumerate(facets):
            pts = points[simplex]
            # Assume triangle for simplicity in 3D hull, but check size
            if len(pts) != 3:
                continue
            a, b, c = pts[0], pts[1], pts[2]
            # Calculate circumcenter
            v1 = a - b
            v2 = a - c
            v = np.cross(v1, v2)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            x = v / norm_v
            
            # Angular radius is arccos(x dot a)
            cos_r = np.clip(np.dot(x, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            
            if r > max_radius:
                max_radius = r
                worst_idx = i
                worst_center = x
        
        if worst_idx == -1:
            return None, None
        return facets[worst_idx], worst_center
    except Exception:
        return None, None


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_circumradius = 0.0
        for simplex in facets:
            pts = points[simplex]
            if len(pts) == 3:
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
                if r > max_circumradius:
                    max_circumradius = r
            else:
                # Fallback for higher dim facets (unlikely in 3D)
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-10:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_r = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    if r > max_circumradius:
                        max_circumradius = r
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf


def _optimize_worst_facet(points, max_time=50.0, step_size=0.05, kick_prob=0.05):
    """
    Optimizes the configuration by repeatedly moving the vertices of the worst facet
    towards its circumcenter.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Pre-compute initial worst facet
    worst_simplex, worst_center = _get_worst_facet(best_points)
    
    while time.time() - start_time < max_time:
        if worst_simplex is None:
            break
            
        # Move vertices of the worst facet towards the circumcenter
        new_points = best_points.copy()
        for idx in worst_simplex:
            p = new_points[idx]
            # Direction towards circumcenter on the tangent plane
            # d = x - (x.p)p
            d = worst_center - np.dot(worst_center, p) * p
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-10:
                d = d / norm_d
                # Move
                new_points[idx] = p + step_size * d
                # Renormalize
                new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
        
        # Evaluate new radius
        new_radius = _covering_radius(new_points)
        
        if new_radius < best_radius:
            best_points = new_points
            best_radius = new_radius
            # Recompute worst facet
            worst_simplex, worst_center = _get_worst_facet(best_points)
            # Decrease step size slightly to refine
            step_size = max(step_size * 0.999, 0.001)
        else:
            # Try a kick to escape local minimum
            if np.random.rand() < kick_prob:
                # Randomly perturb a few points
                k = np.random.randint(1, 5)
                idxs = np.random.choice(N, k, replace=False)
                for idx in idxs:
                    p = best_points[idx]
                    # Random tangent direction
                    rand_vec = np.random.randn(3)
                    d = rand_vec - np.dot(rand_vec, p) * p
                    norm_d = np.linalg.norm(d)
                    if norm_d > 1e-10:
                        d = d / norm_d
                        best_points[idx] = p + 0.1 * d
                        best_points[idx] = best_points[idx] / np.linalg.norm(best_points[idx])
                
                new_radius = _covering_radius(best_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    worst_simplex, worst_center = _get_worst_facet(best_points)
                else:
                    # Revert kick if it didn't help immediately? Or keep exploring?
                    # Let's revert to avoid getting stuck in bad local mins
                    # Actually, if we revert, we are stuck. 
                    # Better to accept small increases? No, we want to minimize.
                    # If kick doesn't help, we are in a local min. 
                    # Let's try a larger kick or reset.
                    pass
            else:
                # Reduce step size
                step_size *= 0.9
                if step_size < 0.001:
                    break
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    np.random.seed(42)
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points
    best_radius = _covering_radius(best_points)
    
    # Try multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random start
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        
        # Optimize from this start
        opt_points, opt_radius = _optimize_worst_facet(rand_points, max_time=5.0)
        
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
    
    # Final optimization from the best point found
    if time.time() < deadline:
        final_points, final_radius = _optimize_worst_facet(best_points, max_time=30.0)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
