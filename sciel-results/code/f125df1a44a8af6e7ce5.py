"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE = 112.0  # Safety margin


def _hull_area(pts):
    """Compute convex hull area using shoelace formula for speed."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Get convex hull indices
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute minimum triangle area for N points using vectorized operations."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            k = np.arange(j + 1, n)
            cross = (x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _param_to_points(params, n_hull, n_inner):
    """
    Convert parameters to point configuration.
    params: [r_hull, theta_offsets..., r_inner, phi_offsets...]
    - r_hull: scale for hull points on a regular n_hull-gon
    - theta_offsets: small angular perturbations for hull points
    - r_inner: scale for inner points
    - phi_offsets: angular offsets for inner points
    """
    n_total = n_hull + n_inner
    points = np.zeros((n_total, 2))
    
    # Hull points: regular n_hull-gon with perturbations
    base_angles = 2 * np.pi * np.arange(n_hull) / n_hull
    theta_off = params[1:1+n_hull]
    angles = base_angles + theta_off
    r = params[0]
    points[:n_hull, 0] = r * np.cos(angles)
    points[:n_hull, 1] = r * np.sin(angles)
    
    # Inner points: arranged in a smaller regular polygon
    if n_inner > 0:
        inner_base_angles = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi / n_inner
        phi_off = params[1+n_hull:1+n_hull+n_inner]
        inner_angles = inner_base_angles + phi_off
        r_inner = params[1+n_hull+n_inner]
        points[n_hull:, 0] = r_inner * np.cos(inner_angles)
        points[n_hull:, 1] = r_inner * np.sin(inner_angles)
    
    return points


def _boundaries_for_params(n_hull, n_inner):
    """Define bounds for parameter optimization."""
    bounds = []
    # r_hull
    bounds.append((0.7, 2.5))
    # theta offsets for hull
    for _ in range(n_hull):
        bounds.append((-0.25, 0.25))
    # phi offsets for inner
    for _ in range(n_inner):
        bounds.append((-0.4, 0.4))
    # r_inner
    if n_inner > 0:
        bounds.append((0.0, 1.2))
    return bounds


def _param_score(params, n_hull, n_inner):
    """Score for parameterized configuration."""
    points = _param_to_points(params, n_hull, n_inner)
    if np.any(~np.isfinite(points)):
        return 0.0
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_a


def _param_neg_score(params, n_hull, n_inner):
    """Negative parameterized score."""
    return -_param_score(params, n_hull, n_inner)


def _refine_with_de(n_hull, n_inner, max_time=12.0):
    """Run differential evolution on parameterized space."""
    start_t = time.time()
    bounds = _boundaries_for_params(n_hull, n_inner)
    
    def obj(params):
        if time.time() - start_t > max_time:
            return 1e10
        return -_param_score(params, n_hull, n_inner)
    
    try:
        result = differential_evolution(
            obj, bounds,
            seed=42,
            maxiter=30,
            tol=1e-5,
            popsize=8,
            polish=False,
            updating='immediate',
            mutation=(0.5, 0.8),
            crossover=0.7
        )
        if result.x is not None:
            points = _param_to_points(result.x, n_hull, n_inner)
            return points
    except Exception:
        pass
    return None


def _local_refine(points, max_time=8.0):
    """Local refinement using Powell method on full point coordinates."""
    start_t = time.time()
    x0 = points.flatten()
    best_pts = points
    best_s = _score(points)
    
    def neg_score_local(x):
        if time.time() - start_t > max_time:
            return 1e10
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    # Try Powell
    try:
        res = minimize(neg_score_local, x0, method='Powell',
                      options={'maxiter': 300, 'xtol': 1e-9, 'ftol': 1e-11})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best_pts = cand
    except Exception:
        pass
    
    # Try Nelder-Mead
    if time.time() - start_t < max_time - 2:
        try:
            res = minimize(neg_score_local, x0, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best_pts = cand
        except Exception:
            pass
    
    return best_pts


def _try_configuration(n_hull, n_inner, time_budget=16.0):
    """Try a specific hull/inner point count configuration."""
    start_t = time.time()
    best_pts = None
    best_score = 0.0
    
    # Start with a good initial guess
    init_pts = _mixed_init(n_hull, n_inner)
    init_score = _score(init_pts)
    if init_score > best_score:
        best_score = init_score
        best_pts = init_pts
    
    # Run differential evolution
    de_time = min(10.0, time_budget - 6)
    if time.time() - start_t < time_budget - 5 and de_time > 2:
        de_pts = _refine_with_de(n_hull, n_inner, max_time=de_time)
        if de_pts is not None:
            s = _score(de_pts)
            if s > best_score:
                best_score = s
                best_pts = de_pts
    
    # Try a few random parameter sets
    rng = np.random.RandomState(123)
    bounds = _boundaries_for_params(n_hull, n_inner)
    n_random = 3
    for _ in range(n_random):
        if time.time() - start_t > time_budget - 4:
            break
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pts = _param_to_points(params, n_hull, n_inner)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Local refinement
    if best_pts is not None and time.time() - start_t < time_budget - 2:
        refine_time = min(6.0, time_budget - 2 - (time.time() - start_t))
        if refine_time > 1:
            refined = _local_refine(best_pts, max_time=refine_time)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_pts = refined
    
    return best_pts, best_score


def _regular_polygon_init(n, r=1.0):
    """Regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _mixed_init(n_hull, n_inner):
    """Mixed hull/inner initialization with optimized ratios."""
    points = _regular_polygon_init(n_hull, 1.0)
    if n_inner > 0:
        # Use golden ratio-like spacing for inner points
        inner_r = 0.618 * 1.0  # Golden ratio conjugate
        inner_angles = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi / n_inner
        inner = inner_r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        points = np.vstack([points, inner])
    return points


def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = 0.0
    
    # Try different hull/inner configurations
    # For 14 points, focus on promising splits
    configs = [
        (14, 0),   # All on hull
        (12, 2),   # 12 hull, 2 inner
        (13, 1),   # 13 hull, 1 inner
        (11, 3),   # 11 hull, 3 inner
    ]
    
    # Time allocation: prioritize early configs
    time_per_config = 14.0
    
    for n_hull, n_inner in configs:
        if time.time() - start_time > DEADLINE - 8:
            break
        
        remaining = DEADLINE - 8 - (time.time() - start_time)
        alloc = min(time_per_config, remaining)
        if alloc < 4:
            break
        
        pts, score = _try_configuration(n_hull, n_inner, time_budget=alloc)
        
        if pts is not None and score > best_score:
            best_score = score
            best_pts = pts
    
    # If we have a good result, do additional refinement
    if best_pts is not None and time.time() - start_time < DEADLINE - 6:
        refine_time = DEADLINE - 6 - (time.time() - start_time)
        refined = _local_refine(best_pts, max_time=refine_time)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_pts = refined
    
    # Final polish with very tight tolerance
    if best_pts is not None and time.time() - start_time < DEADLINE - 3:
        x0 = best_pts.flatten()
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = _regular_polygon_init(N)
    
    return best_pts
# EVOLVE-BLOCK-END
