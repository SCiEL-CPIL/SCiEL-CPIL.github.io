"""Heilbronn 11 triangles: Soft-min optimization with FPS multi-start and Powell polish."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
# There are C(11, 3) = 165 triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _project_to_triangle(pts):
    """
    Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """
    Soft minimum of triangle areas using log-sum-exp.
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _farthest_point_sampling(rng, n_points):
    """
    Generate n_points inside the triangle using Farthest Point Sampling.
    This provides a good initial spread, which is crucial for Heilbronn-type problems.
    """
    points = []
    # Start with a random point near the center
    start_x = 0.5 + (rng.random() - 0.5) * 0.1
    start_y = S3H / 3.0 + (rng.random() - 0.5) * 0.1
    start_pt = _project_to_triangle(np.array([[start_x, start_y]]))[0]
    points.append(start_pt)
    
    for _ in range(n_points - 1):
        best_point = None
        best_min_dist = -1
        # Generate many candidates
        # We use a grid-like or random sampling. Random is faster but less systematic.
        # Let's use a mix of random and grid for robustness.
        candidates = []
        # Random candidates
        for _ in range(200):
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                candidates.append([x, y])
        
        # Grid candidates (coarse)
        # This helps ensure we don't miss "corners" of the configuration space
        # We'll just rely on random for speed, but maybe add a few fixed strategic points?
        # No, let's stick to random but increase count slightly if time permits.
        
        for c in candidates:
            c = _project_to_triangle(np.array([c]))[0]
            # Calculate min distance to existing points
            min_dist = np.inf
            for p in points:
                d = np.linalg.norm(c - p)
                if d < min_dist:
                    min_dist = d
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_point = c
                
        if best_point is not None:
            points.append(best_point)
        else:
            # Fallback
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = S3H * v
                    pt = _project_to_triangle(np.array([[x, y]]))[0]
                    points.append(pt)
                    break
                        
    return np.array(points)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    def _generate_initial_configs():
        """Generate diverse starting points."""
        configs = []
        
        # 1. Symmetric base: Corners, midpoints, centroid, etc.
        base_sym = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [1/6, S3H/6],
            [5/6, S3H/6],
            [1/3, 2*S3H/3],
            [2/3, 2*S3H/3],
        ])
        configs.append(base_sym)

        # 2. Farthest Point Sampling (FPS) - Structural idea from Inspiration 1
        # FPS tends to produce well-spread configurations which are good for maximizing min distance/area.
        for _ in range(5):
            pts = _farthest_point_sampling(rng, N)
            configs.append(pts)
            
        # 3. Perturbed symmetric base
        for _ in range(5):
            pts = base_sym + rng.normal(0, 0.1, size=(N, 2))
            configs.append(_project_to_triangle(pts))

        return configs

    def _optimize_single(pts0, beta_schedule):
        """Run L-BFGS-B with a schedule of beta values."""
        current_pts = pts0.copy()
        
        for beta in beta_schedule:
            if not _elapsed_ok(10):
                break
                
            def objective(x):
                pts = x.reshape(-1, 2)
                pts = _project_to_triangle(pts)
                return -_soft_min_area(pts, beta)
            
            try:
                res = minimize(
                    objective,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                current_pts = res.x.reshape(-1, 2)
                current_pts = _project_to_triangle(current_pts)
                _update_best(current_pts)
            except Exception:
                pass
                
        return current_pts

    # Phase 1: Multi-start optimization with annealing
    configs = _generate_initial_configs()
    
    # Beta schedule: start smooth, get sharp
    beta_schedule = [10, 50, 200, 1000]

    for config in configs:
        if not _elapsed_ok(20):
            break
        _optimize_single(config, beta_schedule)

    # Phase 2: Powell Polish on the best candidate
    # Powell is robust for non-smooth functions and can escape some local minima that gradient-based methods get stuck in.
    # This is the "refinement machinery" from Inspiration 1/2, adapted to our objective.
    if best_pts is not None and _elapsed_ok(15):
        def powell_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            # Add a small penalty if points are too close to boundaries? 
            # _project_to_triangle handles it, but it's non-differentiable.
            # Powell doesn't need gradients, so it handles the projection kinks better than L-BFGS-B might.
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                powell_obj,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 100, 'maxfev': 5000, 'xrtol': 1e-10, 'fatol': 1e-12}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    # Phase 3: Final fine polish with Nelder-Mead on the absolute best
    if best_pts is not None and _elapsed_ok(5):
        def nm_obj_final(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                nm_obj_final,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # Absolute fallback: random points
        pts = rng.random((N, 2))
        for i in range(N):
            u, v = pts[i]
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        best_pts = _project_to_triangle(pts)

    return best_pts
