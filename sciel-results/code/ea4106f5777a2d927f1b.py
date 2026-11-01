import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _hull_radii_and_poles(pts):
    """
    Compute the covering radius and the poles (circumcenters) of all facets.
    Returns:
        radii: array of angular radii for each valid facet
        poles: array of unit vectors pointing to the circumcenters
        faces: the simplices of the convex hull
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Filter out degenerate facets
        valid = area > _EPS
        if not np.any(valid):
            return np.array([np.pi]), np.array([[1.0, 0.0, 0.0]]), faces

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # The pole is the normalized normal vector (for unit sphere, circumcenter is along normal)
        # Actually, for a triangle on the unit sphere, the circumcenter is NOT simply the normal.
        # The circumcenter is the point on the sphere equidistant from A, B, C.
        # It lies on the line passing through the origin and the centroid of the triangle? No.
        # It lies on the normal line.
        # Let P = t * n. We want P @ A = P @ B = P @ C = cos(r).
        # Since n is orthogonal to AB and AC, n @ A = n @ B = n @ C.
        # So P = n / ||n|| * ||n||. Wait, n is already unit.
        # So the circumcenter is indeed the unit normal vector n.
        # Let's verify: n @ A = cos(theta). Since n is unit, the angular distance is arccos(n @ A).
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid]

    except Exception:
        return np.array([np.pi]), np.array([[1.0, 0.0, 0.0]]), np.empty((0, 3), dtype=int)

def _subgradient_descent(pts, deadline, rng):
    """
    Perform subgradient descent by moving the vertices of the worst facet towards its pole.
    """
    best_pts = pts.copy()
    best_r = np.pi
    
    # Initial evaluation
    radii, poles, faces = _hull_radii_and_poles(pts)
    if len(radii) > 0:
        curr_r = np.max(radii)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
    else:
        return best_pts, best_r

    # Adaptive step size
    step = 0.05
    no_improve_count = 0
    
    while time.time() < deadline:
        radii, poles, faces = _hull_radii_and_poles(pts)
        if len(radii) == 0:
            break
            
        curr_r = np.max(radii)
        
        if curr_r < best_r - 1e-9:
            best_r = curr_r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                step *= 0.9
                no_improve_count = 0
                if step < 1e-6:
                    break
        
        # Find worst facet
        worst_idx = np.argmax(radii)
        if radii[worst_idx] < best_r * 0.99: # If we are already better than best, maybe stop or continue carefully
             pass
             
        v_indices = faces[worst_idx]
        pole = poles[worst_idx]
        
        # Move vertices towards pole
        cand = pts.copy()
        for v in v_indices:
            # Vector from point to pole
            dir_vec = pole - pts[v]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > _EPS:
                dir_vec /= dir_norm
            else:
                # If point is at pole, add random noise
                dir_vec = rng.standard_normal(3)
                dir_vec /= (np.linalg.norm(dir_vec) + _EPS)
            
            cand[v] = pts[v] + step * dir_vec
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
        
        # Evaluate candidate
        cand_radii, _, _ = _hull_radii_and_poles(cand)
        if len(cand_radii) > 0:
            cand_r = np.max(cand_radii)
        else:
            cand_r = np.pi
            
        if cand_r < curr_r:
            pts = cand
        else:
            # Try a smaller step if stuck
            step *= 0.8
            
    return best_pts, best_r

def _multi_restart_search(deadline):
    """
    Run multiple restarts of subgradient descent.
    """
    rng = np.random.default_rng(42)
    best_pts = _initial_fib()
    best_r = np.pi
    
    # Evaluate initial fib
    radii, _, _ = _hull_radii_and_poles(best_pts)
    if len(radii) > 0:
        best_r = np.max(radii)
        
    n_restarts = 20
    time_per_restart = (deadline - 5.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
        
        # Run descent for a limited time
        local_deadline = time.time() + time_per_restart * 0.8
        curr_pts, curr_r = _subgradient_descent(pts, local_deadline, rng)
        
        if curr_r < best_r:
            best_r = curr_r
            best_pts = curr_pts.copy()
            
    # Final polish with the best found
    if time.time() < deadline - 2.0:
        best_pts, best_r = _subgradient_descent(best_pts, deadline, rng)
        
    return best_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin
    best_pts = _multi_restart_search(deadline)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
