import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triple indices for vectorized triangle area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles in vectorized form."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    """Negative soft score for minimization."""
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Ellipse parameterization ----------

def _ellipse_params_to_points(params):
    """
    Convert ellipse parameters to points.
    params = [a, b, theta1, theta2, ..., thetaN]
    where a, b are semi-axes, theta_i are angular positions.
    """
    a = params[0]
    b = params[1]
    thetas = params[2:]
    x = a * np.cos(thetas)
    y = b * np.sin(thetas)
    return np.column_stack([x, y])


def _points_to_ellipse_params(points):
    """
    Convert points to ellipse parameters (approximate).
    """
    # Estimate a and b from the bounding box
    a = (max(points[:, 0]) - min(points[:, 0])) / 2
    b = (max(points[:, 1]) - min(points[:, 1])) / 2
    a = max(a, 0.1)
    b = max(b, 0.1)
    thetas = np.arctan2(points[:, 1] / b, points[:, 0] / a)
    return np.concatenate([[a, b], thetas])


def _ellipse_neg_score(params):
    """Negative score for ellipse parameterization."""
    pts = _ellipse_params_to_points(params)
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s


def _ellipse_soft_neg_score(params, temperature):
    """Negative soft score for ellipse parameterization."""
    pts = _ellipse_params_to_points(params)
    s = _soft_score(pts, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Grid-based parameterization ----------

_GRID_3X5 = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
    [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
], dtype=float)

# Which 2 points to remove (indices) - more patterns
_GRID_REMOVALS = [
    (0, 14), (0, 13), (0, 9), (4, 14), (0, 10), (1, 13),
    (4, 10), (2, 12), (1, 4), (9, 14), (0, 4), (10, 14),
    (5, 13), (6, 12), (7, 11), (3, 13), (1, 14), (5, 9),
]


def _grid_to_points(affine_params, removal_idx):
    """
    Convert affine transformation params + removal indices to 13 points.
    affine_params = [a, b, c, d, tx, ty] where:
      x' = a*x + b*y + tx
      y' = c*x + d*y + ty
    """
    a, b, c, d, tx, ty = affine_params
    M = np.array([[a, b], [c, d]])
    t = np.array([tx, ty])
    
    mask = np.ones(15, dtype=bool)
    mask[removal_idx[0]] = False
    mask[removal_idx[1]] = False
    base_pts = _GRID_3X5[mask]
    
    center = np.mean(base_pts, axis=0)
    base_pts_centered = base_pts - center
    
    pts = (base_pts_centered @ M.T) + t
    return pts


def _grid_score(affine_params, removal_idx):
    """Score for grid-based configuration."""
    pts = _grid_to_points(affine_params, removal_idx)
    return _score(pts)


def _neg_grid_score(affine_params, removal_idx):
    """Negative score for minimization."""
    s = _grid_score(affine_params, removal_idx)
    if s <= 0:
        return 1e10
    return -s


# ---------- Seed generators ----------

def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(seed=42):
    """Quasi-regular: regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.03 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_centered_rings():
    """Center + inner ring + outer ring."""
    pts = np.zeros((N, 2))
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[1 + i] = [0.5 * np.cos(a), 0.5 * np.sin(a)]
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[7 + i] = [1.0 * np.cos(a), 1.0 * np.sin(a)]
    return pts


def _seed_golden_angle(seed=0):
    """Points on a circle with golden angle spacing."""
    rng = np.random.RandomState(seed)
    ang = 2 * np.pi * rng.permutation(N) / N
    r = np.ones(N) * (1.0 + 0.02 * rng.randn(N))
    r = np.clip(r, 0.7, 1.3)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_ellipse(a, b, seed=0):
    """Points on an ellipse with given semi-axes."""
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 0.1, N)
    x = a * np.cos(ang)
    y = b * np.sin(ang)
    return np.column_stack([x, y])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(points, deadline, max_temp=0.15, min_temp=1e-7, n_steps=12):
    """
    Anneal the soft-min temperature and optimize at each step.
    Uses L-BFGS-B then Powell at each temperature level.
    """
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # L-BFGS-B on soft objective
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-13},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Powell on soft objective
        if time.time() < deadline - 4:
            try:
                def obj2(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj2, best.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-11, 'fatol': 1e-11},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


def _polish_true_objective(points, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = points.copy()
    best_s = _score(best)
    
    # Powell
    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 350, 'xatol': 1e-12, 'fatol': 1e-12},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Nelder-Mead fine polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Final high-precision Nelder-Mead
    if time.time() < deadline - 1:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _local_search(points, deadline, n_iter=400, sigma_start=0.02, sigma_end=0.0003):
    """
    Coordinate-wise local search with adaptive step size.
    Tries moving each point in random directions, keeping improvements.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break
        
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        for scale in [1.0, 0.5, 0.2, 0.1, 2.0, 0.05]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


def _optimize_ellipse(deadline, a0=1.0, b0=1.0, seed=0):
    """
    Optimize ellipse parameters using differential evolution + local polish.
    """
    start = time.time()
    timeout = start + min(15, deadline - time.time() - 5)
    
    # Differential evolution on ellipse params [a, b, theta1, ..., thetaN]
    bounds = [(0.3, 3.0), (0.3, 3.0)] + [(-np.pi, np.pi)] * N
    
    try:
        de_res = differential_evolution(
            _ellipse_neg_score,
            bounds,
            seed=seed,
            maxiter=30,
            popsize=15,
            tol=1e-8,
            polish=False,
            update_interval=1,
        )
        best_params = de_res.x
        best_pts = _ellipse_params_to_points(best_params)
        best_s = _score(best_pts)
    except Exception:
        # Fallback: just use initial ellipse
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_params = np.concatenate([[a0, b0], thetas])
        best_pts = _ellipse_params_to_points(best_params)
        best_s = _score(best_pts)
    
    # Local polish on raw coordinates
    if time.time() < timeout - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best_pts = cand.copy()
        except Exception:
            pass
    
    return best_pts, best_s


def _optimize_grid_affine(removal_idx, deadline, max_time_per_removal=4.0):
    """
    Optimize the affine transformation for a given grid removal pattern.
    """
    start = time.time()
    timeout = start + max_time_per_removal
    
    bounds = [
        (0.1, 3.0),
        (-1.5, 1.5),
        (-1.5, 1.5),
        (0.1, 3.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
    ]
    
    best_params = None
    best_s = -1
    
    rng = np.random.RandomState(42 + removal_idx[0] * 100 + removal_idx[1])
    
    for trial in range(5):
        if time.time() > timeout:
            break
        
        x0 = np.array([
            rng.uniform(0.5, 2.0),
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.8, 0.8),
            rng.uniform(0.5, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        ])
        
        s = _grid_score(x0, removal_idx)
        if s > best_s:
            best_s = s
            best_params = x0.copy()
    
    if best_params is None:
        return None, -1
    
    # Local polish
    if time.time() < timeout - 1:
        try:
            res = minimize(
                lambda x: _neg_grid_score(x, removal_idx),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            s = _grid_score(res.x, removal_idx)
            if s > best_s:
                best_s = s
                best_params = res.x.copy()
        except Exception:
            pass
    
    if time.time() < timeout - 0.5:
        try:
            res = minimize(
                lambda x: _neg_grid_score(x, removal_idx),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            s = _grid_score(res.x, removal_idx)
            if s > best_s:
                best_s = s
                best_params = res.x.copy()
        except Exception:
            pass
    
    pts = _grid_to_points(best_params, removal_idx)
    return pts, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_points = None
    best_score = -1
    
    # Phase 0: Ellipse optimization (new primary strategy)
    ellipse_configs = [
        (1.0, 1.0, 0),
        (1.2, 0.8, 1),
        (0.8, 1.2, 2),
        (1.5, 0.7, 3),
        (0.7, 1.5, 4),
        (1.3, 1.0, 5),
        (1.0, 1.3, 6),
    ]
    
    ellipse_budget = 40
    ellipse_deadline = start_time + ellipse_budget
    
    for a, b, seed in ellipse_configs:
        if time.time() >= ellipse_deadline - 3:
            break
        try:
            pts, s = _optimize_ellipse(ellipse_deadline, a, b, seed)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 1: Grid-based optimization
    grid_budget = 30
    grid_deadline = start_time + grid_budget + ellipse_budget
    
    for removal in _GRID_REMOVALS:
        if time.time() >= grid_deadline - 2:
            break
        try:
            pts, s = _optimize_grid_affine(removal, grid_deadline, max_time_per_removal=3.0)
            if pts is not None and s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 2: Annealed soft-min optimization on diverse seeds
    inits = []
    inits.append(_seed_regular())
    inits.append(_seed_centered_rings())
    inits.append(_seed_two_ring(7, 0.5, 0.0))
    inits.append(_seed_two_ring(6, 0.4, np.pi / 6))
    inits.append(_seed_two_ring(8, 0.6, np.pi / 4))
    inits.append(_seed_three_ring(5, 4, 4, 0.5, 0.8, 0.0, np.pi / 5))
    inits.append(_seed_three_ring(6, 3, 4, 0.4, 0.7, np.pi / 6, np.pi / 3))
    inits.append(_seed_quasi_regular())
    inits.append(_seed_golden_angle(0))
    inits.append(_seed_golden_angle(7))
    inits.append(_seed_ellipse(1.2, 0.9, 10))
    inits.append(_seed_ellipse(0.9, 1.2, 11))
    
    rng = np.random.RandomState(42)
    for _ in range(3):
        inits.append(rng.random((N, 2)))
    
    if best_points is not None:
        inits.append(best_points.copy())
    
    remaining_budget = deadline - time.time()
    if remaining_budget > 25:
        for i, init in enumerate(inits):
            if time.time() >= deadline - 25:
                break
            try:
                pts, s = _annealed_soft_minimize(init, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 3: Polish the best result
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _polish_true_objective(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 4: Local search refinement
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, s = _local_search(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Final polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
