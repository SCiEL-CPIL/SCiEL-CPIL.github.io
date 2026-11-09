import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _min_triangle_area(points):
    """Vectorized minimum triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    # Compute all triangle areas vectorized
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 0.0

def _score(points):
    """Score: min triangle area / convex hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_a = _min_triangle_area(points)
    if min_a == float('inf'):
        return 0.0
    return min_a / hull_a

def _negative_score(flat_points):
    """Negative score for minimization."""
    pts = flat_points.reshape(N, 2)
    return -_score(pts)

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles) + center[0]
    y = radius * np.sin(angles) + center[1]
    return np.column_stack((x, y))

def _make_12gon_center():
    """12-gon on unit circle + center point."""
    pts = _regular_polygon(12, radius=1.0)
    center = np.array([[0.0, 0.0]])
    return np.vstack([pts, center])

def _push_apart_refinement(points, n_iter=100, step_size=0.05):
    """Iteratively push apart points that form small triangles."""
    pts = points.copy()
    for _ in range(n_iter):
        # Find the minimum area triangle
        min_area = float('inf')
        min_tri = None
        for i, j, k in combinations(range(N), 3):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
            if area < min_area:
                min_area = area
                min_tri = (i, j, k)
        
        if min_tri is None:
            break
            
        i, j, k = min_tri
        p1, p2, p3 = pts[i], pts[j], pts[k]
        
        # Push the points apart from the centroid of the triangle
        centroid = (p1 + p2 + p3) / 3.0
        for idx in [i, j, k]:
            diff = pts[idx] - centroid
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                pts[idx] += step_size * diff / norm
            else:
                # Random nudge if at centroid
                pts[idx] += step_size * np.random.randn(2) * 0.1
        
        # Decay step size
        step_size *= 0.95
    
    return pts

def _optimize_from_start(start_pts, max_iter=3000, deadline=None):
    """Optimize from a starting configuration using Nelder-Mead."""
    try:
        res = minimize(
            _negative_score,
            start_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if deadline is not None and time.time() >= deadline:
            return None
        return res.x.reshape(N, 2)
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with 12-gon + center (better topology for N=13)
    initial_points = _make_12gon_center()
    best_score = _score(initial_points)
    best_points = initial_points.copy()
    
    # Also try regular 13-gon as alternative
    reg13 = _regular_polygon(13, radius=1.0)
    s13 = _score(reg13)
    if s13 > best_score:
        best_score = s13
        best_points = reg13.copy()
    
    # Optimize from best initial
    if time.time() < deadline - 5:
        opt = _optimize_from_start(best_points, max_iter=5000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    # Multi-restart with different strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Perturbed 12-gon + center
    for trial in range(5):
        if time.time() >= deadline - 10:
            break
        noise_scale = 0.05 + trial * 0.03
        start_pts = _make_12gon_center() + rng.normal(0, noise_scale, (N, 2))
        opt = _optimize_from_start(start_pts, max_iter=2000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    # Strategy 2: Push-apart refinement on current best
    if time.time() < deadline - 15:
        refined = _push_apart_refinement(best_points, n_iter=200, step_size=0.03)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
        
        # Optimize from refined
        if time.time() < deadline - 5:
            opt = _optimize_from_start(best_points, max_iter=3000, deadline=deadline)
            if opt is not None:
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_points = opt.copy()
    
    # Strategy 3: Random restarts with larger perturbations
    for trial in range(3):
        if time.time() >= deadline - 10:
            break
        start_pts = best_points + rng.normal(0, 0.15 + trial * 0.05, (N, 2))
        opt = _optimize_from_start(start_pts, max_iter=2000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    # Final refinement push
    if time.time() < deadline - 5:
        refined = _push_apart_refinement(best_points, n_iter=100, step_size=0.02)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
        
        opt = _optimize_from_start(best_points, max_iter=2000, deadline=deadline)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt.copy()
    
    return best_points
