import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude self
    mask = np.ones(len(pts), dtype=bool)
    # Find self index robustly
    self_idx = np.argmin(dists)
    mask[self_idx] = False
    d = np.sort(dists[mask])
    # d has 19 elements
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(20):
        for j in range(i+1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return False
    # Check convexity: all cross products same sign
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i+1) % 20]
        p2 = pts[(i+2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _support_to_pts(h):
    """
    Convert support function values h[0..19] to vertex coordinates.
    h[i] is the distance from origin to edge i.
    Edge i has normal angle alpha_i = i * pi/10 + pi/20.
    The vertices are intersections of consecutive edge lines.
    """
    n = len(h)
    alphas = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/n
    
    # Line i: x cos(alpha_i) + y sin(alpha_i) = h[i]
    # Vertex i is intersection of line i-1 and line i (indices mod n)
    # Let's use vertex i as intersection of edge i and edge i+1? 
    # Standard convention: vertex i is between edge i-1 and edge i.
    # Let's define edge i normal as alpha_i.
    # Vertex i is intersection of edge i-1 and edge i.
    
    pts = np.zeros((n, 2))
    for i in range(n):
        i_prev = (i - 1) % n
        a1 = alphas[i_prev]
        a2 = alphas[i]
        h1 = h[i_prev]
        h2 = h[i]
        
        # x cos(a1) + y sin(a1) = h1
        # x cos(a2) + y sin(a2) = h2
        
        # Matrix form:
        # [cos(a1) sin(a1)] [x]   [h1]
        # [cos(a2) sin(a2)] [y] = [h2]
        
        A = np.array([
            [np.cos(a1), np.sin(a1)],
            [np.cos(a2), np.sin(a2)]
        ])
        b = np.array([h1, h2])
        
        try:
            pts[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Should not happen for distinct normals
            pts[i] = np.array([0, 0])
            
    return pts

def _objective(params):
    """
    Objective to minimize. Returns negative of the score.
    params: h values (20 floats)
    """
    # Ensure h is positive to guarantee convexity containing origin?
    # Actually, for a convex polygon, h must be such that the intersection is bounded and non-empty.
    # For a regular polygon, h is constant.
    # If we allow h to vary, we must ensure the polygon is strictly convex.
    # The support function representation guarantees convexity if the polygon is the intersection of the halfplanes.
    # However, if h[i] is too small or negative, the intersection might be empty or unbounded?
    # No, intersection of halfplanes is always convex. It is bounded if the normals span the circle.
    # With 20 equally spaced normals, it is always bounded.
    # But we need strict convexity: no collinear vertices.
    # If h[i] == h[i+1] and the angles are 180 deg apart? No, angles are 18 deg apart.
    # Strict convexity fails if three consecutive vertices are collinear.
    # This happens if the "turn" is zero.
    # In support function, this corresponds to specific relations.
    # To be safe, we penalize invalid configurations.
    
    h = params
    pts = _support_to_pts(h)
    
    # Check validity
    if not _is_valid(pts):
        return 1e6  # Large penalty
    
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e6
        
    score = _min_vertex_score(norm_pts)
    
    # We want to maximize score, so minimize -score
    # Also, we might want to regularize h to be smooth (close to constant) to avoid weird shapes?
    # The task doesn't require smoothness, just convexity.
    # But "equidistant" in the name suggests we want distances to be equal-ish.
    # The score function rewards uniformity of distances.
    
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    n = 20
    # Initial guess: Regular 20-gon
    # For regular 20-gon, all h are equal.
    # Let's calculate h for a regular 20-gon of circumradius R=1.
    # Distance from center to side = R * cos(pi/n)
    h_reg = np.cos(np.pi / n)
    h_init = np.full(n, h_reg)
    
    best_score = -1e9
    best_pts = None
    
    # Get initial points
    pts_init = _support_to_pts(h_init)
    norm_pts_init, valid = _normalize(pts_init)
    if valid and _is_valid(norm_pts_init):
        score_init = _min_vertex_score(norm_pts_init)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts_init.copy()
    else:
        # Fallback to regular polygon directly if support conversion fails (shouldn't happen)
        base_angles = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        norm_pts_init, valid = _normalize(best_pts)
        if valid:
            best_score = _min_vertex_score(norm_pts_init)

    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    
    # Number of starts
    num_starts = 10
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        # Perturb initial h
        if i == 0:
            h_start = h_init.copy()
        else:
            # Random perturbation
            # Keep h positive to ensure origin is inside? Not strictly necessary for validity, 
            # but helps stability. The polygon is defined by halfplanes x.n <= h.
            # If h is negative, the halfplane is far away.
            # Let's perturb around h_reg.
            h_start = h_init + rng.normal(0, 0.1, size=n)
            h_start = np.maximum(h_start, 0.1) # Ensure positive to avoid weirdness
            
        res = minimize(
            _objective, 
            h_start, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
        )
        
        if time.time() > deadline:
            break
            
        h_opt = res.x
        pts_opt = _support_to_pts(h_opt)
        
        if not _is_valid(pts_opt):
            continue
            
        norm_pts_opt, valid = _normalize(pts_opt)
        if not valid:
            continue
            
        score_opt = _min_vertex_score(norm_pts_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt.copy()
            
    # Final check and return
    if best_pts is None:
        # Absolute fallback
        base_angles = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        
    return best_pts
