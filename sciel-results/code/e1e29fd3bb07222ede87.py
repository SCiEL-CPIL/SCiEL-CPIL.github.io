"""Heilbronn 11 triangles: Model-and-optimize with soft-min and projection."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 140  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
# There are C(11, 3) = 165 triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _project_to_triangle(pts):
    """
    Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    This is a hard constraint enforcement.
    """
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Clamp y to [0, S3H]
        y = max(0.0, min(S3H, y))
        # For a given y, x must be between the left and right boundaries.
        # Left boundary: line from (0,0) to (0.5, S3H) -> y = sqrt(3)x -> x = y/sqrt(3)
        # Right boundary: line from (1,0) to (0.5, S3H) -> y = -sqrt(3)(x-1) -> x = 1 - y/sqrt(3)
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    Returns the raw area (not normalized).
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
    This is differentiable and approximates the hard minimum as beta -> infinity.
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
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid numerical overflow/underflow, we shift by the max area
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # If beta is very large, exp(shifted) can be 0 for most terms, only max term remains 1
    # log(sum(exp)) is stable
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        # Ensure points are inside
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

        # 2. Random uniform points
        for _ in range(5):
            pts = rng.random((N, 2))
            # Map to triangle
            for i in range(N):
                u, v = pts[i]
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u + 0.5 * v
                y = S3H * v
                pts[i] = [x, y]
            configs.append(_project_to_triangle(pts))
            
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
    # Area is small (~0.01), so beta needs to be large enough to distinguish areas.
    # If beta=10, difference of 0.01 in area -> exp(-0.1) vs exp(0). Not very sharp.
    # If beta=100, difference 0.01 -> exp(-1) vs 1. Better.
    # If beta=1000, difference 0.01 -> exp(-10) vs 1. Very sharp.
    beta_schedule = [10, 50, 200, 1000]

    for config in configs:
        if not _elapsed_ok(20):
            break
        _optimize_single(config, beta_schedule)

    # Phase 2: Nelder-Mead Polish on the best candidate
    if best_pts is not None and _elapsed_ok(10):
        # Try a few NM restarts around the best point with decreasing step size
        for scale in [0.05, 0.02, 0.01]:
            if not _elapsed_ok(5):
                break
            for _ in range(3):
                if not _elapsed_ok(5):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)
                
                def nm_obj(x):
                    p = x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    return -_min_triangle_area(p)
                
                try:
                    res = minimize(
                        nm_obj,
                        cand.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    p = res.x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    _update_best(p)
                except Exception:
                    pass

    # Phase 3: Final fine polish on the absolute best
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
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
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

# EVOLVE-BLOCK-END
