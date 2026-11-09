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
    # r1 is fixed to 1.0
    r1 = 1.0
    pts = np.zeros((13, 2))
    # Center
    pts[0] = [0, 0]
    
    # Inner Hexagon
    for i in range(6):
        angle = i * np.pi / 3.0
        pts[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
    # Outer Hexagon
    for i in range(6):
        angle = i * np.pi / 3.0 + theta
        pts[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
    return pts

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 580 # Safety margin
    
    # 1. Generate candidate initial points
    candidates = []
    
    # Candidate 1: Structured (Hexagons) - optimized locally first? 
    # Let's just add the raw structured one and the regular polygon.
    # We will optimize each from scratch.
    
    # The structured one from previous best
    # We can try to recover the optimized structured one quickly or just use the raw one.
    # The raw structured one (r2=2, theta=pi/6) is a decent start.
    candidates.append(_generate_structured_points(2.0, np.pi / 6.0))
    
    # Candidate 2: Regular 13-gon
    candidates.append(_generate_regular_polygon(13))
    
    # Candidate 3: Regular 13-gon with slight perturbation
    np.random.seed(42)
    reg_poly = _generate_regular_polygon(13)
    reg_poly += 0.1 * np.random.normal(0, 1, reg_poly.shape)
    candidates.append(reg_poly)
    
    best_pts = candidates[0]
    best_score = _score(best_pts)
    
    # 2. Optimize each candidate
    for start_pts in candidates:
        if time.time() > deadline:
            break
            
        def _obj(x):
            pts = x.reshape(13, 2)
            return -_score(pts)
        
        x_start = start_pts.flatten()
        
        try:
            # Use Nelder-Mead with tight tolerances
            res = minimize(_obj, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
            
            if res.fun < -best_score:
                pts_opt = res.x.reshape(13, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass
            
    # 3. Final refinement with a smaller step size if time permits
    if time.time() < deadline - 60:
        def _obj(x):
            pts = x.reshape(13, 2)
            return -_score(pts)
            
        x_start = best_pts.flatten()
        try:
            res = minimize(_obj, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500})
            if res.fun < -best_score:
                pts_opt = res.x.reshape(13, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass

    if _score(best_pts) <= 0:
        best_pts = _generate_regular_polygon(13)

    return best_pts
