# EVOLVE-BLOCK-START
import numpy as np
import time


def _vertex_score(polygon: np.ndarray) -> np.ndarray:
    """
    Compute the per-vertex score for a (20, 2) polygon.
    Returns an array of shape (20,) with values in [0, 1].
    """
    n = polygon.shape[0]
    # Pairwise distances
    diffs = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.sort(dists[i])
        # d is shape (19,)
        if np.any(d < 0.05):
            continue
        
        gaps = d[1:] - d[:-1]
        k = np.argmin(gaps)
        # The gap is between d[k] and d[k+1]
        D = (d[k] + d[k+1]) / 2.0
        if D < 0.1:
            continue
        
        if k == 0:
            idx = np.array([0, 1, 2, 3])
        elif k == 17:
            idx = np.array([15, 16, 17, 18])
        else:
            idx = np.array([k-1, k, k+1, k+2])
        
        vals = d[idx]
        raw = np.mean(np.maximum(vals / D, D / vals))
        scores[i] = min(1.0, 1.0 / raw)
    return scores


def _is_strictly_convex(polygon: np.ndarray, tol=1e-10) -> bool:
    """
    Check if the polygon is strictly convex.
    1. No repeated points.
    2. All turns have the same non-zero sign.
    3. Total turning is +/- 2*pi (implied by 1 & 2 for simple polygons, but let's be safe).
    """
    n = polygon.shape[0]
    # Check repeated points
    dists = np.sum((polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :])**2, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < tol):
        return False
    
    # Compute cross products of consecutive edges
    edges = np.roll(polygon, -1, axis=0) - polygon
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    # All cross products must be non-zero and same sign
    if np.any(np.abs(cross) < tol):
        return False
    if np.any(np.diff(np.sign(cross)) != 0):
        return False
        
    # Total turning check (optional but good)
    # For a closed convex polygon, the sum of exterior angles is 2*pi.
    # We can check the winding number or just trust the cross product check for simplicity 
    # as long as it's a simple polygon. The cross product check ensures local convexity.
    # A locally convex simple polygon is globally convex.
    return True


def _total_score(polygon: np.ndarray) -> float:
    """
    Normalize, check convexity, return min vertex score.
    """
    # Normalize: subtract centroid
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    
    # Mean distance to centroid
    dists = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # Check convexity
    if not _is_strictly_convex(p):
        return 0.0
    
    # Compute scores
    scores = _vertex_score(p)
    return np.min(scores)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices maximizing the minimum vertex score.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # 1. Generate initial candidates
    candidates = []
    
    # A. Regular 20-gon
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    p_reg = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    candidates.append(p_reg)
    
    # B. Clusters of 5 (Pentagons) at 4 corners
    # Centers at 0, pi/2, pi, 3pi/2
    centers_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    R_large = 10.0
    R_small = 1.0
    for c_ang in centers_angles:
        cx = R_large * np.cos(c_ang)
        cy = R_large * np.sin(c_ang)
        # Pentagon vertices
        # Orient pentagon so one vertex points outward? Or inward?
        # Let's try pointing a vertex outward first.
        # Angle offset for pentagon vertices
        pent_angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        # Rotate so one vertex is at angle c_ang
        pent_angles = pent_angles + c_ang
        px = cx + R_small * np.cos(pent_angles)
        py = cy + R_small * np.sin(pent_angles)
        pts = np.stack([px, py], axis=1)
        candidates.append(pts)
    
    # Combine into one polygon? 
    # The union of these points needs to be in cyclic order.
    # Let's collect all points, sort by angle around origin, and check convexity.
    all_pts = np.vstack(candidates)
    # Sort by angle
    angles = np.arctan2(all_pts[:, 1], all_pts[:, 0])
    idx = np.argsort(angles)
    p_cluster = all_pts[idx]
    
    # Check convexity. If not convex, we might need to adjust R_large/R_small.
    # For now, add it. The optimizer will likely reject it if invalid (score 0) 
    # or we can skip adding if invalid.
    # Actually, if it's not convex, score is 0. Let's try to make it convex.
    # With R_large=10, R_small=1, the pentagons are well separated.
    # The hull will be the convex hull of the 20 points.
    # If we just return the points in angular order, are they strictly convex?
    # The "flat" parts between clusters might have points that are not on the hull?
    # No, all 20 points are on the hull if the clusters are "bumpy" enough.
    # A regular pentagon is convex. If we place 4 convex shapes far apart, their convex hull is the union of their convex hulls connected by tangents.
    # The vertices of the pentagons will be on the hull.
    # So p_cluster should be convex.
    if _is_strictly_convex(p_cluster):
        candidates.append(p_cluster)

    # C. Other patterns?
    # Maybe 2 clusters of 10?
    # Or 5 clusters of 4?
    # Let's stick with these for now to save time.
    
    # 2. Evaluate candidates
    best_poly = None
    best_score = -1.0
    for p in candidates:
        if time.time() > deadline - 30.0:
            break
        s = _total_score(p)
        if s > best_score:
            best_score = s
            best_poly = p.copy()
            
    # 3. Local Optimization (Nelder-Mead)
    if best_poly is None:
        # Fallback
        return p_reg
    
    # We need to optimize the parameters that define the polygon.
    # For the regular polygon, there are no free parameters (shape is fixed).
    # For the cluster polygon, we have:
    # - R_large (distance of cluster centers)
    # - R_small (radius of clusters)
    # - Pentagon rotation (maybe)
    # - Number of points per cluster? (Fixed at 5)
    # - Cluster angular positions? (Fixed at 90 deg apart)
    
    # Let's define a parameter vector for the cluster design.
    # params = [R_large, R_small, rotation_offset]
    # Actually, let's allow the cluster centers to move slightly?
    # No, symmetry is likely optimal.
    
    def objective_cluster(params):
        R_large = params[0]
        R_small = params[1]
        rot_offset = params[2]
        
        if R_large < 1.0 or R_small < 0.1 or R_small > R_large:
            return -1.0
            
        pts = []
        centers_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        for c_ang in centers_angles:
            cx = R_large * np.cos(c_ang)
            cy = R_large * np.sin(c_ang)
            pent_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + c_ang + rot_offset
            px = cx + R_small * np.cos(pent_angles)
            py = cy + R_small * np.sin(pent_angles)
            pts.append(np.stack([px, py], axis=1))
        
        all_pts = np.vstack(pts)
        angles = np.arctan2(all_pts[:, 1], all_pts[:, 0])
        idx = np.argsort(angles)
        p = all_pts[idx]
        
        if not _is_strictly_convex(p):
            return -1.0
            
        return _total_score(p)

    # Initialize params
    # Start from the evaluated cluster candidate
    # We need to recover R_large, R_small from p_cluster? 
    # We used 10.0 and 1.0.
    x0 = np.array([10.0, 1.0, 0.0])
    
    # Check if we have time
    if time.time() < deadline - 60.0:
        from scipy.optimize import minimize
        
        # Use Nelder-Mead
        # Maximize score -> Minimize -score
        res = minimize(lambda x: -objective_cluster(x), x0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'initial_simplex': [1.0, 0.1, 0.1]})
        
        if time.time() < deadline - 30.0 and res.x is not None:
            s_opt = -res.fun
            if s_opt > best_score:
                # Reconstruct polygon
                R_large, R_small, rot_offset = res.x
                pts = []
                centers_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                for c_ang in centers_angles:
                    cx = R_large * np.cos(c_ang)
                    cy = R_large * np.sin(c_ang)
                    pent_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + c_ang + rot_offset
                    px = cx + R_small * np.cos(pent_angles)
                    py = cy + R_small * np.sin(pent_angles)
                    pts.append(np.stack([px, py], axis=1))
                all_pts = np.vstack(pts)
                angles = np.arctan2(all_pts[:, 1], all_pts[:, 0])
                idx = np.argsort(angles)
                p_opt = all_pts[idx]
                
                if _is_strictly_convex(p_opt):
                    best_poly = p_opt
                    best_score = s_opt

    # 4. Try optimizing the regular polygon?
    # The regular polygon is a local maximum? 
    # Let's try perturbing it slightly.
    # Parameters: deviations from regular angles? 
    # This is high dimensional (19 dof). 
    # Maybe just stick with the cluster optimization which is low dimensional.
    
    # 5. Final check
    if best_poly is None:
        return p_reg
        
    return best_poly


# EVOLVE-BLOCK-END
