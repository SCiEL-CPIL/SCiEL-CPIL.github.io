import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _score_configuration(points):
    """
    Calculate the score for a given set of points.
    Score = min_triangle_area / convex_hull_area
    """
    points = np.array(points, dtype=float)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
        
    # Vectorized triangle area calculation
    # Get all combinations of 3 indices
    # For n=13, there are 286 triangles.
    # We can compute all areas at once using broadcasting or explicit loops.
    # Explicit loop is fine for n=13, but let's try to make it fast.
    
    min_area = float('inf')
    
    # Pre-compute differences? 
    # area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    
    # Using itertools.combinations is clean
    from itertools import combinations
    idx = range(n)
    for i, j, k in combinations(idx, 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Cross product magnitude
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p1[0]
        dy2 = p3[1] - p1[1]
        
        area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
        if area < min_area:
            min_area = area
            if min_area < 1e-12:
                return 0.0

    return min_area / hull_area

def _generate_concentric_points(n1, r1, phi1, n2, r2, phi2, include_center=True):
    """
    Generate points for a configuration with up to 2 concentric rings and an optional center.
    Total points must be 13.
    If include_center: 1 + n1 + n2 = 13 => n1 + n2 = 12
    If not include_center: n1 + n2 = 13
    """
    points = []
    if include_center:
        points.append([0.0, 0.0])
    
    if n1 > 0:
        angles1 = np.arange(n1) * (2 * np.pi / n1) + phi1
        pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        points.extend(pts1)
        
    if n2 > 0:
        angles2 = np.arange(n2) * (2 * np.pi / n2) + phi2
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        points.extend(pts2)
        
    return np.array(points)

def construct():
    best_score = 0.0
    best_points = None
    
    # We search over topologies (n1, n2) such that 1 + n1 + n2 = 13
    # So n1 + n2 = 12.
    # Possible pairs (n1, n2) where n1, n2 >= 0:
    # (0, 12), (1, 11), ..., (12, 0)
    
    topologies = []
    for n1 in range(0, 13):
        n2 = 12 - n1
        topologies.append((n1, n2))
    
    candidates = []
    
    # Grid parameters
    # r in [0.1, 1.0]
    # phi in [0, 2*pi/n)
    
    # We will perform a coarse grid search first
    r_vals = np.linspace(0.1, 1.0, 20)
    
    for n1, n2 in topologies:
        # If n1==0, phi1 is irrelevant. If n2==0, phi2 is irrelevant.
        # We can fix phi1=0 always.
        # If n2 > 0, we search phi2. If n2 == 0, we set phi2=0.
        
        if n2 > 0:
            phi2_vals = np.linspace(0.0, 2 * np.pi / n2, 20)
        else:
            phi2_vals = [0.0]
            
        if n1 > 0:
            # phi1 is fixed to 0 by symmetry
            pass
            
        for r1 in r_vals:
            if n1 == 0:
                r1 = 0.0 # Doesn't matter
            for r2 in r_vals:
                if n2 == 0:
                    r2 = 0.0
                for phi2 in phi2_vals:
                    # If n1==0, r1 is ignored in generator?
                    # _generate_concentric_points checks if n1>0.
                    points = _generate_concentric_points(n1, r1, 0.0, n2, r2, phi2, include_center=True)
                    score = _score_configuration(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                    
                    # Store top candidates for local optimization
                    if score > 0.05:
                        candidates.append((score, [r1, r2, phi2], n1, n2))
                        
    # Local optimization on the best few candidates
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 50 candidates to ensure we find the global max
    # Deduplicate roughly by checking if params are very close?
    # For simplicity, just take top 20 distinct ones.
    
    top_candidates = candidates[:20]
    
    for score, params, n1, n2 in top_candidates:
        # params: [r1, r2, phi2]
        
        def objective(params):
            r1, r2, phi2 = params
            # Bound check
            if r1 < 0.01 or r2 < 0.01:
                # If n1==0, r1 can be 0. If n2==0, r2 can be 0.
                # But the generator ignores 0-count rings.
                # However, the optimizer might drift.
                # Penalty if radius is negative or too small for a non-empty ring.
                if (n1 > 0 and r1 < 0.01) or (n2 > 0 and r2 < 0.01):
                    return 10.0
            points = _generate_concentric_points(n1, r1, 0.0, n2, r2, phi2, include_center=True)
            s = _score_configuration(points)
            return -s
        
        x0 = np.array(params)
        
        # Define bounds
        bounds = []
        if n1 > 0:
            bounds.append((0.01, 1.0))
        else:
            bounds.append((0.0, 0.0)) # Fixed
        if n2 > 0:
            bounds.append((0.01, 1.0))
        else:
            bounds.append((0.0, 0.0))
        
        if n2 > 0:
            bounds.append((0.0, 2 * np.pi / n2))
        else:
            bounds.append((0.0, 0.0))
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            r1_opt, r2_opt, phi2_opt = res.x
            points_opt = _generate_concentric_points(n1, r1_opt, 0.0, n2, r2_opt, phi2_opt, include_center=True)
            score_opt = _score_configuration(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = np.arange(13) * (2 * np.pi / 13)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
