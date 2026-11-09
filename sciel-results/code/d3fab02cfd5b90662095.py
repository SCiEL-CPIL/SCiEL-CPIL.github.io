# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _vertex_score(v, pts):
    """Compute the vertex score for a single vertex v given all points."""
    # Distances to all other vertices
    diffs = pts - v
    dists = np.linalg.norm(diffs, axis=1)
    # Exclude self (should be 0, but be safe)
    mask = np.arange(len(pts)) != np.where(np.all(pts == v, axis=1))[0][0] if np.any(np.all(pts == v, axis=1)) else np.ones(len(pts), dtype=bool)
    # Actually, simpler: just take the 19 non-zero distances
    dists_sorted = np.sort(dists)
    # The first element is 0 (distance to self), so take the rest
    d = dists_sorted[1:]  # 19 distances
    
    if len(d) != 19:
        return 0.0
    
    # Check if any distance < 0.05
    if np.any(d < 0.05):
        return 0.0
    
    # Find k: first index minimizing d[k+1] - d[k]
    diffs_d = d[1:] - d[:-1]  # 18 differences
    k = np.argmin(diffs_d)
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    # Choose four distances
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    
    raw = np.mean(np.maximum(four_d / D, D / four_d))
    return min(1.0, 1.0 / raw)


def _min_vertex_score(pts):
    """Compute the minimum vertex score over all vertices."""
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return np.min(scores)


def _is_strictly_convex(pts):
    """Check if points form a strictly convex polygon in cyclic order."""
    n = len(pts)
    if n < 3:
        return False
    
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return False
    
    # Compute cross products of consecutive edges
    cross_products = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        p2 = pts[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    
    # All cross products must be nonzero and of the same sign
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    
    if np.all(cross_products > 0) or np.all(cross_products < 0):
        return True
    
    return False


def _normalize(pts):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    mean_dist = np.mean(np.linalg.norm(pts_centered, axis=1))
    if mean_dist <= 0.01:
        return None
    return pts_centered / mean_dist


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum vertex score.
    Uses optimization with a smooth surrogate objective.
    """
    start_time = time.time()
    deadline = 600.0
    margin = 30.0
    
    n = 20
    
    # Start with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    pts0 = np.column_stack([x0, y0])
    
    best_pts = pts0.copy()
    best_score = _min_vertex_score(_normalize(best_pts))
    
    # Objective function for optimization (maximize min vertex score)
    def objective(params):
        if time.time() > deadline - margin:
            return 0.0
        
        pts = np.column_stack([params[:n], params[n:2*n]])
        
        # Check convexity cheaply
        if not _is_strictly_convex(pts):
            return -1.0
        
        norm_pts = _normalize(pts)
        if norm_pts is None:
            return -1.0
        
        return _min_vertex_score(norm_pts)
    
    # Try L-BFGS-B optimization
    try:
        result = minimize(
            lambda p: -objective(p),
            pts0.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.fun < 0:
            score_candidate = -result.fun
        else:
            score_candidate = 0.0
        
        if score_candidate > best_score:
            best_pts = np.column_stack([result.x[:n], result.x[n:2*n]])
            best_score = score_candidate
            
    except Exception:
        pass
    
    # Try differential evolution for global exploration
    if time.time() < deadline - margin:
        try:
            bounds = [(-2, 2)] * (2 * n)
            
            def de_objective(params):
                if time.time() > deadline - margin:
                    return 0.0
                return -objective(params)
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=100,
                tol=1e-8,
                seed=42,
                polish=False
            )
            
            if de_result.fun < 0:
                score_candidate = -de_result.fun
            else:
                score_candidate = 0.0
            
            if score_candidate > best_score:
                best_pts = np.column_stack([de_result.x[:n], de_result.x[n:2*n]])
                best_score = score_candidate
                
        except Exception:
            pass
    
    # Polish with L-BFGS-B from best found
    if time.time() < deadline - margin:
        try:
            result2 = minimize(
                lambda p: -objective(p),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            
            if result2.fun < 0:
                score_candidate = -result2.fun
            else:
                score_candidate = 0.0
            
            if score_candidate > best_score:
                best_pts = np.column_stack([result2.x[:n], result2.x[n:2*n]])
                best_score = score_candidate
                
        except Exception:
            pass
    
    # Final normalization and validation
    norm_pts = _normalize(best_pts)
    if norm_pts is None:
        # Fallback to regular polygon
        norm_pts = pts0.copy()
    
    return norm_pts


# EVOLVE-BLOCK-END
