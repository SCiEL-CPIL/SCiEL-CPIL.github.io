import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    i, j, k = i.T.ravel(), j.T.ravel(), k.T.ravel()
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
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

def _min_triangle_area(points):
    """Return the minimum triangle area, or 0 if degenerate."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        areas = _triangle_areas(points)
        return float(np.min(areas))
    except Exception:
        return 0.0

def _coordinate_descent(points, max_iters=200, tol=1e-8):
    """Coordinate descent with exact line search to maximize min triangle area."""
    pts = points.copy()
    n = len(pts)
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for coord in range(2):
                # Exact line search along coordinate i, coord
                def obj(t):
                    new_pts = pts.copy()
                    new_pts[i, coord] = t
                    return -_min_triangle_area(new_pts)
                
                try:
                    res = minimize_scalar(obj, bounds=(-2.0, 2.0), method='bounded', 
                                          options={'xatol': 1e-6, 'maxiter': 100})
                    if res.success and res.fun < obj(pts[i, coord]) - tol:
                        pts[i, coord] = res.x
                        improved = True
                except Exception:
                    pass
        
        if not improved:
            break
    
    return pts

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 580  # Safety margin
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    # Simulated annealing with coordinate descent refinement
    rng = np.random.default_rng(42)
    current_points = best_points.copy()
    current_score = best_score
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    sa_iter = 0
    max_sa_iters = 50000
    
    while time.time() < deadline - 30 and sa_iter < max_sa_iters:
        # Perturbation: move 1-3 random points
        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(n, size=n_perturb, replace=False)
        perturbations = rng.normal(0, 0.1, size=(n_perturb, 2))
        
        new_points = current_points.copy()
        new_points[idxs] += perturbations
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score or (T > T_min and rng.random() < np.exp((new_score - current_score) / T)):
            current_points = new_points
            current_score = new_score
            
            # Periodically do coordinate descent
            if sa_iter % 50 == 0:
                cd_points = _coordinate_descent(current_points, max_iters=10)
                cd_score = _score(cd_points)
                if cd_score > current_score:
                    current_points = cd_points
                    current_score = cd_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Reject, but still occasionally try coordinate descent from current
            if sa_iter % 200 == 0:
                cd_points = _coordinate_descent(current_points, max_iters=15)
                cd_score = _score(cd_points)
                if cd_score > best_score:
                    best_score = cd_score
                    best_points = cd_points.copy()
        
        T *= cooling
        sa_iter += 1
    
    # Final coordinate descent polish
    if time.time() < deadline - 10:
        final_points = _coordinate_descent(best_points, max_iters=50)
        final_score = _score(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Fallback
    if _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
