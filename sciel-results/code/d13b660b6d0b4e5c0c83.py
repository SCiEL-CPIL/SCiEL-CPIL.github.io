"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE = 115.0  # Safety margin


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
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
    for i in range(n_hull):
        points[i, 0] = r * np.cos(angles[i])
        points[i, 1] = r * np.sin(angles[i])
    
    # Inner points: arranged in a smaller regular polygon
    if n_inner > 0:
        inner_base_angles = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi / n_inner
        phi_off = params[1+n_hull:1+n_hull+n_inner]
        inner_angles = inner_base_angles + phi_off
        r_inner = params[1+n_hull+n_inner]
        for i in range(n_inner):
            idx = n_hull + i
            points[idx, 0] = r_inner * np.cos(inner_angles[i])
            points[idx, 1] = r_inner * np.sin(inner_angles[i])
    
    return points


def _boundaries_for_params(n_hull, n_inner):
    """Define bounds for parameter optimization."""
    bounds = []
    # r_hull
    bounds.append((0.5, 3.0))
    # theta offsets for hull
    for _ in range(n_hull):
        bounds.append((-0.3, 0.3))
    # phi offsets for inner
    for _ in range(n_inner):
        bounds.append((-0.3, 0.3))
    # r_inner
    if n_inner > 0:
        bounds.append((0.0, 1.5))
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


def _refine_with_de(n_hull, n_inner, max_time=15.0):
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
            maxiter=50,
            tol=1e-6,
            popsize=10,
            polish=False,
            updating='deferred'
        )
        if result.x is not None:
            points = _param_to_points(result.x, n_hull, n_inner)
            return points
    except Exception:
        pass
    return None


def _local_refine(points, max_time=10.0):
    """Local refinement using Powell method on full point coordinates."""
    start_t = time.time()
    x0 = points.flatten()
    
    def neg_score_local(x):
        if time.time() - start_t > max_time:
            return 1e10
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    try:
        res = minimize(neg_score_local, x0, method='Powell',
                      options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10})
        if res.x is not None:
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Try Nelder-Mead as fallback
    try:
        res = minimize(neg_score_local, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                              'adaptive': True})
        if res.x is not None:
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points


def _try_configuration(n_hull, n_inner, time_budget=20.0):
    """Try a specific hull/inner point count configuration."""
    start_t = time.time()
    best_pts = None
    best_score = 0.0
    
    # Run differential evolution
    if time.time() - start_t < time_budget - 5:
        de_pts = _refine_with_de(n_hull, n_inner, max_time=time_budget - 10)
        if de_pts is not None:
            s = _score(de_pts)
            if s > best_score:
                best_score = s
                best_pts = de_pts
    
    # Also try a few random parameter sets
    rng = np.random.RandomState(123)
    bounds = _boundaries_for_params(n_hull, n_inner)
    for _ in range(5):
        if time.time() - start_t > time_budget - 5:
            break
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pts = _param_to_points(params, n_hull, n_inner)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Local refinement
    if best_pts is not None and time.time() - start_t < time_budget - 3:
        refined = _local_refine(best_pts, max_time=time_budget - 3)
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
    """Mixed hull/inner initialization."""
    points = _regular_polygon_init(n_hull, 1.0)
    if n_inner > 0:
        inner_angles = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi / n_inner
        inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        points = np.vstack([points, inner])
    return points


def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = 0.0
    
    # Try different hull/inner configurations
    # For 14 points, try various splits
    configs = [
        (14, 0),   # All on hull
        (13, 1),   # 13 hull, 1 inner
        (12, 2),   # 12 hull, 2 inner
        (11, 3),   # 11 hull, 3 inner
        (10, 4),   # 10 hull, 4 inner
    ]
    
    # Time allocation: give more time to promising configs
    time_per_config = 18.0
    
    for n_hull, n_inner in configs:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        remaining = DEADLINE - 5 - (time.time() - start_time)
        alloc = min(time_per_config, remaining)
        
        pts, score = _try_configuration(n_hull, n_inner, time_budget=alloc)
        
        if pts is not None and score > best_score:
            best_score = score
            best_pts = pts
    
    # If we have a good result, do additional refinement
    if best_pts is not None and time.time() - start_time < DEADLINE - 8:
        refined = _local_refine(best_pts, max_time=DEADLINE - 8 - (time.time() - start_time))
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_pts = refined
    
    # Final polish with very tight tolerance
    if best_pts is not None and time.time() - start_time < DEADLINE - 3:
        x0 = best_pts.flatten()
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
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
