"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 115.0  # seconds wall-clock limit; keep safety margin


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))


def _hull_area(points):
    """Compute the convex hull area of the point set."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, volume is area
    except Exception:
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return _min_triangle_area(points) / hull_area


def _negative_score(points):
    return -_score(points)


def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    # Strategy: 12-gon + center topology
    # Place 12 points on a regular 12-gon (or perturbed) and 1 point near center
    # This utilizes interior space better than regular 13-gon

    def _regular_12_plus_center(perturb=0.0, center_offset=0.0, rng_seed=42):
        rng = np.random.RandomState(rng_seed)
        # 12 points on a circle of radius 1
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        if perturb > 0:
            pts += rng.randn(12, 2) * perturb
        # Center point
        center = np.array([center_offset, center_offset])
        all_pts = np.vstack([pts, center])
        return all_pts

    # Initial best: regular 12-gon + center
    init_pts = _regular_12_plus_center(perturb=0.0, center_offset=0.0)
    init_score = _score(init_pts)
    best_points = init_pts.copy()
    best_score = init_score

    # Try different center offsets and perturbations
    rng = np.random.RandomState(12345)

    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Local optimization using scipy
    def _objective(x):
        pts = x.reshape(-1, 2)
        # Penalize if hull area is too small
        hull_area = _hull_area(pts)
        if hull_area < 1e-6:
            return 1e10
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area

    # Try multiple initial configurations
    configs = []
    
    # Config 1: Regular 12-gon + center
    configs.append(_regular_12_plus_center(perturb=0.0, center_offset=0.0, rng_seed=42))
    
    # Config 2: Regular 12-gon + slightly offset center
    for dx in [-0.1, 0.1, -0.05, 0.05]:
        for dy in [-0.1, 0.1, -0.05, 0.05]:
            configs.append(_regular_12_plus_center(perturb=0.0, center_offset=dx, rng_seed=42 + int(abs(dx*100) + abs(dy*100))))
    
    # Config 3: Perturbed 12-gon + center
    for perturb in [0.01, 0.02, 0.03, 0.05]:
        for seed in [1, 2, 3, 4, 5]:
            configs.append(_regular_12_plus_center(perturb=perturb, center_offset=0.0, rng_seed=seed))
    
    # Config 4: Random near 12-gon
    for seed in range(20):
        rng2 = np.random.RandomState(seed)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        radii = 1.0 + rng2.randn(12) * 0.05
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        center = rng2.randn(2) * 0.05
        all_pts = np.vstack([pts, center])
        configs.append(all_pts)

    # Evaluate all configs
    for cfg in configs:
        if _time_remaining() < 5.0:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()

    # Local optimization on best config
    if _time_remaining() > 10.0:
        x0 = best_points.flatten()
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # More local optimizations from different starting points
    for cfg in configs[:10]:
        if _time_remaining() < 15.0:
            break
        x0 = cfg.flatten()
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Final check and return
    if _time_remaining() < 1.0:
        return best_points

    # One more round of refinement
    if _time_remaining() > 5.0:
        x0 = best_points.flatten()
        try:
            result = minimize(_objective, x0, method='Powell',
                             options={'maxiter': 2000, 'ftol': 1e-10})
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    return best_points
