"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized computation of areas for all triples is complex to write efficiently 
    # for small N in pure numpy without loops, but 13^3 is small.
    # However, we can speed it up by precomputing differences.
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            for k in range(j + 1, n):
                v2 = points[k] - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback for degenerate cases (collinear points)
        if len(points) < 3:
            return 0.0
        # If ConvexHull fails, it's likely collinear or duplicate points
        # Calculate area of the polygon formed by sorting points by angle around centroid
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        # Shoelace formula
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        n = len(sorted_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j] - x[j] * y[i]
        return abs(area) * 0.5

def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _regular_polygon(n):
    """Generate vertices of a regular n-gon on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Start with regular polygon as initial guess
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Optimization setup
    # We want to maximize score, so minimize -score
    def objective(x):
        pts = x.reshape(-1, 2)
        # Small penalty if points become collinear or hull area vanishes
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-6:
            return 1e10
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area

    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Optimize the regular polygon (full 2D freedom)
    # 2. If time permits, try radial optimization (13 vars)
    # 3. If time permits, try multi-start with perturbations

    # Run 1: Full 2D optimization from regular polygon
    if time.time() < deadline - 10:
        res = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'initial_simplex': 0.1})
        if res.success or res.fun < -best_score:
            candidate = res.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate

    # Run 2: Radial optimization
    # Parameterize points as r_i * (cos(theta_i), sin(theta_i))
    # theta_i are fixed from regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    def objective_radial(r):
        pts = np.column_stack((r * cos_a, r * sin_a))
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-6:
            return 1e10
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area

    if time.time() < deadline - 10:
        # Start with all radii = 1
        r0 = np.ones(N)
        res_r = minimize(objective_radial, r0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
        if res_r.success or res_r.fun < -best_score:
            r_opt = res_r.x
            candidate = np.column_stack((r_opt * cos_a, r_opt * sin_a))
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate

    # Run 3: Perturbed regular polygon (multi-start)
    # Try a few random perturbations to escape local minima
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Perturb the best points found so far
        noise = rng.normal(0, 0.05, size=best_points.shape)
        points0 = best_points + noise
        
        def objective_perturbed(x):
            pts = x.reshape(-1, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-6:
                return 1e10
            min_area = _min_triangle_area(pts)
            return -min_area / hull_area

        res_p = minimize(objective_perturbed, points0.flatten(), method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res_p.success or res_p.fun < -best_score:
            candidate = res_p.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate

    return best_points

# EVOLVE-BLOCK-END
