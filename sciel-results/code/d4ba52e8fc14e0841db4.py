"""Heilbronn convex regions n=14 — multi-ring exploration with Nelder-Mead refinement."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area using scipy.spatial.ConvexHull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: compute via shoelace on sorted angles
        n = len(points)
        if n < 3:
            return 0.0
        cx, cy = points[:, 0].mean(), points[:, 1].mean()
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        order = np.argsort(angles)
        hpts = points[order]
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hpts[i][0] * hpts[j][1] - hpts[j][0] * hpts[i][1]
        return abs(area) * 0.5

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points):
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _ring_config(n_out, n_in, r, theta):
    """Generate points: n_out on outer ring (radius 1), n_in on inner ring (radius r)."""
    if n_out <= 0:
        angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
        return r * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    if n_in <= 0:
        angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        return np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _optimize_local(points, maxiter=500, time_limit=30.0):
    """Local optimization using Nelder-Mead with time guard."""
    start_time = time.time()
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
    
    # Second pass with tighter tolerance if time remains
    if time.time() - start_time < time_limit * 0.7:
        try:
            x0 = best_pts.flatten()
            res2 = minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': maxiter // 2, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() - start_time < time_limit:
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_pts = res2.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    return best_score, best_pts

def _optimize_angles_on_circle(angles, time_limit=20.0):
    """Optimize angles of points on a unit circle to maximize score."""
    start_time = time.time()
    
    def obj(angles_flat):
        a = np.sort(angles_flat)
        a = a - a[0]
        pts = np.column_stack([np.cos(a), np.sin(a)])
        return -_score(pts)
    
    a0 = np.sort(angles)
    a0 = a0 - a0[0]
    best_score = _score(np.column_stack([np.cos(a0), np.sin(a0)]))
    best_angles = a0.copy()
    
    try:
        res = minimize(
            obj, a0, method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_angles = np.sort(res.x)
                best_angles = best_angles - best_angles[0]
    except Exception:
        pass
    
    pts = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    return best_score, pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy 1: Explore diverse ring configurations with local Nelder-Mead refinement
    ring_configs = [
        # (n_out, n_in, r, theta)
        (7, 7, 0.5, 0.0),
        (7, 7, 0.6, np.pi / 7),
        (7, 7, 0.7, np.pi / 7),
        (7, 7, 0.8, np.pi / 7),
        (8, 6, 0.5, 0.0),
        (8, 6, 0.6, np.pi / 12),
        (8, 6, 0.7, np.pi / 12),
        (6, 8, 0.5, 0.0),
        (6, 8, 0.6, np.pi / 24),
        (6, 8, 0.7, np.pi / 24),
        (9, 5, 0.5, 0.0),
        (9, 5, 0.6, np.pi / 45),
        (5, 9, 0.5, 0.0),
        (5, 9, 0.6, np.pi / 90),
        (10, 4, 0.5, 0.0),
        (10, 4, 0.6, np.pi / 20),
        (4, 10, 0.5, 0.0),
        (4, 10, 0.6, np.pi / 40),
        (11, 3, 0.5, 0.0),
        (3, 11, 0.5, 0.0),
        (12, 2, 0.5, 0.0),
        (2, 12, 0.5, 0.0),
    ]
    
    # Sort by heuristic: prefer balanced splits
    def heuristic(config):
        n_out, n_in, r, theta = config
        balance = min(n_out, n_in) / max(n_out, n_in)
        return -balance
    
    ring_configs.sort(key=heuristic)
    
    time_per_config = 6.0
    for n_out, n_in, r, theta in ring_configs:
        if time.time() > deadline - 15:
            break
        
        points = _ring_config(n_out, n_in, r, theta)
        score, opt_pts = _optimize_local(points, maxiter=200, time_limit=time_per_config)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = opt_pts.copy()
    
    # Strategy 2: Optimize angles on a single unit circle
    if time.time() < deadline - 25:
        # Evenly spaced
        angles_even = np.linspace(0, 2 * np.pi, N, endpoint=False)
        score, pts = _optimize_angles_on_circle(angles_even, time_limit=15.0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = pts.copy()
        
        # Perturbed evenly spaced
        if time.time() < deadline - 15:
            np.random.seed(42)
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                angles_pert = angles_even + np.random.randn(N) * 0.15
                score, pts = _optimize_angles_on_circle(angles_pert, time_limit=10.0)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_points = pts.copy()
    
    # Strategy 3: Differential evolution over free 2D positions (short run)
    if time.time() < deadline - 30:
        def neg_score_free(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            return -_score(pts)
        
        x0 = best_overall_points.flatten().tolist() if best_overall_points is not None else \
             _regular_polygon(N).flatten().tolist()
        bounds = [(-2, 2)] * (2 * N)
        try:
            res = differential_evolution(
                neg_score_free,
                bounds,
                x0=x0,
                maxiter=30,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=False,
                timeout=25
            )
            if time.time() > deadline:
                pass
            else:
                s = -res.fun
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = res.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Strategy 4: Final local polish on best candidate
    if time.time() < deadline - 10 and best_overall_points is not None:
        score, opt_pts = _optimize_local(best_overall_points, maxiter=300, time_limit=15.0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = opt_pts.copy()
    
    # Fallback
    if best_overall_points is None:
        best_overall_points = _regular_polygon(N, 1.0)
    
    # Final verification
    final_score = _score(best_overall_points)
    if final_score <= 0:
        return _regular_polygon(N, 1.0)
    
    return best_overall_points
