import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    n = len(points)
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _generate_structured_points(r2, theta):
    r1 = 1.0
    pts = np.zeros((13, 2))
    pts[0] = [0, 0]
    for i in range(6):
        angle = i * np.pi / 3.0
        pts[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3.0 + theta
        pts[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    return pts

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _geometric_ascent(points, deadline, max_iter=2000):
    """
    Geometric gradient-free ascent that directly targets the degenerate triangle.
    Instead of random-walking in parameter space, this method identifies the minimum-area
    triangle and moves its vertices to maximize that area while minimizing hull area.
    """
    pts = points.copy()
    best_score = _score(pts)
    if best_score <= 0:
        return pts, best_score
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find the minimum area triangle
        areas = _triangle_areas(pts)
        min_idx = np.argmin(areas)
        
        # Get the indices of the three points forming the min triangle
        i, j, k = [], [], []
        count = 0
        for a in range(13):
            for b in range(a + 1, 13):
                for c in range(b + 1, 13):
                    if count == min_idx:
                        i, j, k = a, b, c
                        break
                    count += 1
                if count == min_idx:
                    break
            if count == min_idx:
                break
        
        if not i or not j or not k:
            i, j, k = 0, 1, 2
        
        # Calculate the centroid of the three points
        centroid = (pts[i] + pts[j] + pts[k]) / 3.0
        
        # Calculate the normal vector to the triangle (for 2D, this gives us the direction)
        # The area is proportional to the cross product magnitude
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        if abs(cross) < 1e-10:
            # Points are nearly collinear, perturb randomly
            np.random.seed(iteration)
            pts[i] += 0.01 * np.random.normal(0, 1, 2)
            pts[j] += 0.01 * np.random.normal(0, 1, 2)
            pts[k] += 0.01 * np.random.normal(0, 1, 2)
            continue
        
        # The gradient of area with respect to each point
        # For triangle (p1, p2, p3), area = 0.5 * |cross(p2-p1, p3-p1)|
        # d(area)/dp1 = -0.5 * cross_unit
        # d(area)/dp2 = 0.5 * cross_unit_perp
        # d(area)/dp3 = -0.5 * cross_unit_perp
        
        # Normalized cross product direction
        cross_norm = np.array([v2[1], -v2[0]])  # Perpendicular to v2
        cross_norm /= np.linalg.norm(cross_norm) if np.linalg.norm(cross_norm) > 0 else 1
        
        # Move points to increase the area of this triangle
        # Move i in direction that increases area, j and k similarly
        step_size = 0.05
        
        # The area increases when we move points apart in the perpendicular direction
        pts[i] -= step_size * cross_norm
        pts[j] += step_size * cross_norm
        pts[k] -= step_size * cross_norm
        
        # Also try to reduce hull area by pulling convex hull points inward
        # First, identify which points are on the convex hull
        try:
            hull = ConvexHull(pts)
            hull_indices = set(hull.vertices)
        except:
            hull_indices = set(range(13))
        
        # For points on the convex hull that are NOT in the min triangle,
        # try to move them slightly toward the centroid to reduce hull area
        all_centroid = np.mean(pts, axis=0)
        for idx in hull_indices:
            if idx not in [i, j, k]:
                # Move slightly toward center
                pts[idx] = 0.995 * pts[idx] + 0.005 * all_centroid
        
        # Check if this improved the score
        new_score = _score(pts)
        if new_score > best_score:
            best_score = new_score
        else:
            # Revert and try a different perturbation
            pts = points.copy() if 'points' in locals() else pts
            # Actually, we need to keep the best version
            # Let's just continue with the current best
            break
    
    return pts, best_score

def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 580  # Safety margin
    
    # 1. Generate candidate initial points
    candidates = []
    
    # Candidate 1: Structured (Hexagons)
    candidates.append(_generate_structured_points(2.0, np.pi / 6.0))
    
    # Candidate 2: Regular 13-gon
    candidates.append(_generate_regular_polygon(13))
    
    # Candidate 3: Regular 13-gon with slight perturbation
    np.random.seed(42)
    reg_poly = _generate_regular_polygon(13)
    reg_poly += 0.1 * np.random.normal(0, 1, reg_poly.shape)
    candidates.append(reg_poly)
    
    # Candidate 4: Another structured variant
    candidates.append(_generate_structured_points(1.8, np.pi / 12.0))
    
    # Candidate 5: Concentric circles with different radii
    pts_circ = np.zeros((13, 2))
    pts_circ[0] = [0, 0]
    for i in range(6):
        angle = i * np.pi / 3.0
        pts_circ[i+1] = [0.7 * np.cos(angle), 0.7 * np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3.0 + np.pi / 6
        pts_circ[i+7] = [1.5 * np.cos(angle), 1.5 * np.sin(angle)]
    candidates.append(pts_circ)
    
    best_pts = candidates[0]
    best_score = _score(best_pts)
    
    # 2. Optimize each candidate using geometric ascent
    for start_pts in candidates:
        if time.time() > deadline:
            break
        
        # First, do a quick Nelder-Mead to get to a decent local optimum
        def _obj(x):
            pts = x.reshape(13, 2)
            return -_score(pts)
        
        x_start = start_pts.flatten()
        
        try:
            res = minimize(_obj, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
            
            if res.fun < -best_score:
                pts_opt = res.x.reshape(13, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass
        
        # Then, apply geometric ascent to the best result
        if time.time() > deadline - 60:
            break
        
        pts_geometric, score_geometric = _geometric_ascent(best_pts, deadline, max_iter=1000)
        if score_geometric > best_score:
            best_score = score_geometric
            best_pts = pts_geometric
    
    # 3. Final refinement with Nelder-Mead
    if time.time() < deadline - 30:
        def _obj(x):
            pts = x.reshape(13, 2)
            return -_score(pts)
            
        x_start = best_pts.flatten()
        try:
            res = minimize(_obj, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300})
            if res.fun < -best_score:
                pts_opt = res.x.reshape(13, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass
    
    # 4. Final geometric ascent pass
    if time.time() < deadline - 10:
        pts_final, score_final = _geometric_ascent(best_pts, deadline, max_iter=500)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final

    if _score(best_pts) <= 0:
        best_pts = _generate_regular_polygon(13)

    return best_pts
