"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _triangle_areas(pts):
    """Compute all triangle areas for N points. Returns array of size C(N,3)."""
    n = len(pts)
    # Use cross product: area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # Vectorized: for all i < j < k
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                areas[i, j, k] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    # Extract upper triangular indices
    idx = np.triu_indices(n, k=2)
    return areas[idx]


def _min_triangle_area(pts):
    """Minimum triangle area among all triples."""
    areas = _triangle_areas(pts)
    return np.min(areas)


def _convex_hull_area(pts):
    """Area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, .volume gives area
    except Exception:
        return 0.0


def _score(pts):
    """Min triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _regular_n_gon(n, radius=1.0, angle_offset=0.0):
    """Vertices of regular n-gon inscribed in circle of given radius."""
    angles = angle_offset + 2 * np.pi * np.arange(n) / n
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def construct():
    deadline = time.time() + 110  # 110s to be safe within 120s limit
    
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Regular 14-gon as initial guess
    pts_reg = _regular_n_gon(N)
    s = _score(pts_reg)
    if s > best_score:
        best_score = s
        best_pts = pts_reg.copy()
    
    # Strategy 2: Two concentric regular polygons (7-gon + 7-gon) with optimized radii and rotation
    # This is a structured family that may outperform regular 14-gon
    def two_polygon_score(params):
        r1 = params[0]  # radius of outer 7-gon
        r2 = params[1]  # radius of inner 7-gon
        rot = params[2]  # rotation of inner relative to outer
        outer = _regular_n_gon(7, radius=r1, angle_offset=0.0)
        inner = _regular_n_gon(7, radius=r2, angle_offset=rot)
        pts = np.vstack([outer, inner])
        return -_score(pts)
    
    if time.time() < deadline:
        # Optimize the two-polygon configuration
        bounds = [(0.1, 1.5), (0.1, 1.5), (0, 2 * np.pi / 7)]
        res = differential_evolution(
            two_polygon_score, bounds,
            seed=42, maxiter=100, tol=1e-8,
            popsize=15, polish=True
        )
        if res.fun is not None and not np.isnan(res.fun):
            r1, r2, rot = res.x
            outer = _regular_n_gon(7, radius=r1, angle_offset=0.0)
            inner = _regular_n_gon(7, radius=r2, angle_offset=rot)
            pts = np.vstack([outer, inner])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    # Strategy 3: Full local optimization from best starting point
    if best_pts is not None and time.time() < deadline:
        # Fix p0 at origin, p1 at (1,0) to remove translation/rotation invariance
        # Actually, let's keep all points free but use a penalty-free approach
        # We'll optimize all 28 coordinates
        
        def objective(x):
            pts = x.reshape(N, 2)
            # Center the points
            centroid = np.mean(pts, axis=0)
            pts_centered = pts - centroid
            s = _score(pts_centered)
            return -s
        
        x0 = best_pts.flatten()
        
        # Use L-BFGS-B with bounds to keep points in a reasonable region
        bounds = [(-5, 5)] * (2 * N)
        
        res = minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.fun is not None and not np.isnan(res.fun):
            pts_opt = res.x.reshape(N, 2)
            centroid = np.mean(pts_opt, axis=0)
            pts_opt = pts_opt - centroid
            s = _score(pts_opt)
            if s > best_score:
                best_score = s
                best_pts = pts_opt.copy()
    
    # Strategy 4: Try multiple random restarts with local optimization if time allows
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            # Random points in unit disk
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.3, 1.0, N)
            pts_rand = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            
            def objective(x):
                pts = x.reshape(N, 2)
                centroid = np.mean(pts, axis=0)
                pts_c = pts - centroid
                s = _score(pts_c)
                return -s
            
            x0 = pts_rand.flatten()
            bounds = [(-3, 3)] * (2 * N)
            res = minimize(
                objective, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            if res.fun is not None and not np.isnan(res.fun):
                pts_opt = res.x.reshape(N, 2)
                centroid = np.mean(pts_opt, axis=0)
                pts_opt = pts_opt - centroid
                s = _score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_pts = pts_opt.copy()
    
    # Final: normalize to unit area convex hull
    if best_pts is None:
        best_pts = _regular_n_gon(N)
    
    # Normalize: scale so convex hull area = 1
    hull_area = _convex_hull_area(best_pts)
    if hull_area > 1e-12:
        scale = 1.0 / np.sqrt(hull_area)
        best_pts = best_pts * scale
    
    return best_pts
