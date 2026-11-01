"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas(pts):
    """Compute areas of all triples of points."""
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas[i, j, k] = area
    return areas

def _min_triangle_area(pts):
    """Return the minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Return the area of the convex hull of pts."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area via shoelace formula on hull points
        # If ConvexHull fails, use all points (shouldn't happen for valid convex config)
        from scipy.spatial import ConvexHull as CH
        try:
            h = CH(pts)
            return h.volume
        except:
            # Last resort: approximate
            return 1.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(pts)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return min_ta / hull_a

def _negative_score(pts):
    return -_score(pts)

def _regular_polygon(n, scale=1.0):
    """Regular n-gon centered at origin with circumradius scale."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([scale * np.cos(angles), scale * np.sin(angles)])
    return pts

def _optimize_from_initial(pts_init, max_iter=5000, time_limit=100):
    """Use L-BFGS-B to optimize point positions to maximize min triangle area / hull area."""
    # We'll optimize all 2N coordinates
    # To handle the scale invariance, we can fix one point and use relative positions,
    # but L-BFGS-B works better with a proper objective. Let's just optimize all coords
    # and rely on the objective function.
    
    x0 = pts_init.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return _negative_score(pts)
    
    start_time = time.time()
    best_x = x0.copy()
    best_val = objective(x0)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 100, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            break
    
    return best_x.reshape(-1, 2), -best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Start with regular polygon and optimize
    # Regular 14-gon
    pts_reg = _regular_polygon(N)
    
    # Scale doesn't matter for the ratio, but let's keep it reasonable
    # Optimize from regular polygon
    try:
        opt_pts, opt_score = _optimize_from_initial(pts_reg, max_iter=3000, time_limit=120)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_pts is not None:
            return best_pts
        return pts_reg
    
    # Strategy 2: Try different initializations
    # Points on a circle with slightly perturbed radii
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Create initial points on circle with some radial variation
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + 0.1 * np.sin(2 * angles + trial * 0.5)
        pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        try:
            opt_pts, opt_score = _optimize_from_initial(pts_init, max_iter=2000, time_limit=20)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass
    
    # Strategy 3: Try a "two-ring" configuration
    # 7 points on inner ring, 7 on outer ring
    for trial in range(3):
        if time.time() > deadline:
            break
        
        pts_init = np.zeros((N, 2))
        inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        
        inner_r = 0.5 + 0.05 * trial
        outer_r = 1.0
        
        for i in range(7):
            pts_init[2 * i] = [inner_r * np.cos(inner_angles[i]), inner_r * np.sin(inner_angles[i])]
            pts_init[2 * i + 1] = [outer_r * np.cos(outer_angles[i]), outer_r * np.sin(outer_angles[i])]
        
        try:
            opt_pts, opt_score = _optimize_from_initial(pts_init, max_iter=2000, time_limit=15)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass
    
    # Strategy 4: Random restarts with local optimization
    np.random.seed(42)
    for trial in range(10):
        if time.time() > deadline:
            break
        
        pts_init = np.random.randn(N, 2) * 0.5
        # Make them roughly convex by projecting to convex position
        # Simple approach: sort by angle from centroid
        centroid = pts_init.mean(axis=0)
        angles = np.arctan2(pts_init[:, 1] - centroid[1], pts_init[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        pts_init = pts_init[sorted_idx]
        
        try:
            opt_pts, opt_score = _optimize_from_initial(pts_init, max_iter=1000, time_limit=8)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: regular polygon
    return pts_reg
# EVOLVE-BLOCK-END
