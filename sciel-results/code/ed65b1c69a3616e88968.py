import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def _triangle_areas(points):
    n = len(points)
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
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

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # 1. Initial Candidates
    candidates = []
    
    # Regular 13-gon
    candidates.append(_generate_regular_polygon(13))
    
    # Structured: Center + 2 Hexagons (various parameters)
    candidates.append(_generate_structured_points(2.0, np.pi / 6.0))
    candidates.append(_generate_structured_points(1.5, np.pi / 6.0))
    candidates.append(_generate_structured_points(2.5, np.pi / 6.0))
    candidates.append(_generate_structured_points(2.0, 0.0))
    candidates.append(_generate_structured_points(2.0, np.pi / 12.0))
    
    # Random candidates (normalized to unit square roughly)
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.uniform(-1, 1, size=(13, 2))
        # Ensure not all collinear or degenerate
        candidates.append(pts)

    best_pts = candidates[0]
    best_score = _score(best_pts)
    
    for pts in candidates[1:]:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    # 2. Global Optimization: Short Differential Evolution
    # We run DE for a limited time to find a promising basin.
    # Since DE is slow, we run it for a few seconds and take the best result.
    
    def _obj(x):
        pts = x.reshape(13, 2)
        return -_score(pts)
    
    bounds = [(-3, 3)] * 26
    
    # We will run DE with a callback to stop if time is running out,
    # but DE in scipy doesn't have a clean time-based stopper in older versions easily.
    # Instead, we limit maxiter and popsize to ensure it finishes quickly.
    # Let's allocate ~30s to DE.
    
    de_start = time.time()
    try:
        # Conservative settings to ensure it returns within time
        # 50 iterations * 10 pop * 26 dims = 13000 evals.
        # If 1ms/eval, 13s.
        res = differential_evolution(
            _obj, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6, 
            seed=42,
            init='latinhypercube',
            polish=False # We will do our own polish
        )
        
        if res.fun < -best_score:
            pts_de = res.x.reshape(13, 2)
            score_de = _score(pts_de)
            if score_de > best_score:
                best_score = score_de
                best_pts = pts_de
                
    except Exception:
        pass
    
    # 3. Local Refinement: Multi-start Nelder-Mead
    # We will try to refine the current best point and a few other good candidates.
    # Nelder-Mead is fast and good for local polishing.
    
    # Generate a list of starting points for NM
    nm_starts = [best_pts]
    
    # Add a few random perturbations of the best point
    for i in range(5):
        if time.time() > deadline - 10:
            break
        noise = rng.normal(0, 0.5, size=(13, 2))
        pt_perturbed = best_pts + noise
        nm_starts.append(pt_perturbed)
        
    # Add the DE result if it was different and good
    if 'res' in locals() and res.fun < -best_score + 0.001:
         nm_starts.append(res.x.reshape(13, 2))

    # Also try refining the initial structured points if they are close to best
    for pts in candidates[:3]:
         if _score(pts) > best_score * 0.9:
             nm_starts.append(pts)

    for start_pt in nm_starts:
        if time.time() > deadline - 5:
            break
            
        try:
            res_nm = minimize(
                _obj, 
                start_pt.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < -best_score:
                pts_nm = res_nm.x.reshape(13, 2)
                score_nm = _score(pts_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_pts = pts_nm
                    
        except Exception:
            pass

    # 4. Final Check
    if _score(best_pts) <= 0:
        best_pts = _generate_regular_polygon(13)

    return best_pts
