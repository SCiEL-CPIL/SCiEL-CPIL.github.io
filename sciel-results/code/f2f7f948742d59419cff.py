import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 120.0
SAFETY_MARGIN = 8.0
START_TIME = time.time()

# Precompute all triangle index triples for vectorized min-area
_TRIPLE_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)], dtype=np.intp)

def _min_triangle_area(points):
    """Vectorized minimum triangle area over all C(N,3) triples."""
    p_i = points[_TRIPLE_IDX[:, 0]]
    p_j = points[_TRIPLE_IDX[:, 1]]
    p_k = points[_TRIPLE_IDX[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Objective to minimize: -min_triangle_area / hull_area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -min_area / hull_area

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _concentric(ring_sizes, radii, phase=0.0):
    """Points on concentric circles with rotational symmetry."""
    pts = []
    for size, r in zip(ring_sizes, radii):
        if size == 0:
            continue
        ang = np.linspace(0, 2 * np.pi, size, endpoint=False) + phase
        pts.append(np.column_stack((r * np.cos(ang), r * np.sin(ang))))
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)

def _normalize(points):
    """Scale so convex hull area = 1 (scale invariance)."""
    a = _convex_hull_area(points)
    if a < 1e-12:
        return points
    return points * np.sqrt(1.0 / a)

def _local_polish(x0, start_time):
    """Nelder-Mead local refinement with time guard."""
    try:
        res = minimize(
            _score, x0, method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        if res.fun < _score(x0):
            return res.x
    except Exception:
        pass
    return x0

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
    Returns negative soft-min score for minimization.
    """
    try:
        hull = ConvexHull(points)
        area_hull = hull.volume
    except Exception:
        return 1e6
        
    if area_hull < 1e-9:
        return 1e6

    p_i = points[_TRIPLE_IDX[:, 0]]
    p_j = points[_TRIPLE_IDX[:, 1]]
    p_k = points[_TRIPLE_IDX[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    areas = 0.5 * cross

    # Soft-min via log-sum-exp: -log(sum(exp(-beta * a_i))) / beta approximates min(a_i)
    # For stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta

    score = soft_min_area / area_hull
    return -score

def _soft_objective(points_flat, beta=50.0):
    """Soft-min objective for gradient-based optimization."""
    points = points_flat.reshape(-1, 2)
    return _soft_min_score(points, beta)

def construct():
    start_time = time.time()
    best_points = None
    best_score = -float('inf')

    def consider(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()

    # ---- Diverse initial configurations ----
    inits = []

    # Regular 13-gon
    inits.append(_regular_polygon(N))

    # Concentric ring arrangements (rotational symmetry breaks small-triangle issue)
    ring_configs = [
        ([1, 12], [0.0, 1.0]),
        ([5, 8], [0.0, 1.0]),
        ([4, 9], [0.0, 1.0]),
        ([7, 6], [0.0, 1.0]),
        ([6, 7], [0.0, 1.0]),
        ([3, 5, 5], [0.0, 0.55, 1.0]),
        ([2, 5, 6], [0.0, 0.5, 1.0]),
        ([1, 5, 7], [0.0, 0.5, 1.0]),
        ([4, 4, 5], [0.0, 0.55, 1.0]),
        ([1, 4, 8], [0.0, 0.5, 1.0]),
        ([2, 3, 8], [0.0, 0.45, 1.0]),
        ([1, 6, 6], [0.0, 0.55, 1.0]),
    ]
    for sizes, radii in ring_configs:
        for phase in (0.0, np.pi / 13.0):
            inits.append(_concentric(sizes, radii, phase=phase))

    # Jittered versions of the best ring configs to escape symmetries
    for sizes, radii in ring_configs[:6]:
        base = _concentric(sizes, radii)
        jittered = base + np.random.normal(0, 0.08, base.shape)
        inits.append(jittered)

    # ---- Phase 1: Local polish from each diverse start ----
    for x0 in [init.flatten() for init in inits]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        x_polished = _local_polish(x0, start_time)
        pts = x_polished.reshape(-1, 2)
        consider(pts)

    # ---- Phase 2: Gradient-based refinement using soft-min surrogate ----
    # The soft-min makes the objective smooth, allowing L-BFGS to navigate better
    if best_points is not None:
        current = best_points.copy()
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            try:
                result = minimize(
                    lambda x: _soft_objective(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized = result.x.reshape(-1, 2)
                consider(optimized)
                # Keep the optimized point for the next beta (annealing)
                current = optimized
            except Exception:
                pass

    # ---- Phase 3: Second round of local polish from the best, with tiny perturbations ----
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        for _ in range(8):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            pert = best_points + np.random.normal(0, 0.03, best_points.shape)
            x_polished = _local_polish(pert.flatten(), start_time)
            pts = x_polished.reshape(-1, 2)
            consider(pts)

    # ---- Final: normalize hull area to 1 (scale invariance, cleaner output) ----
    if best_points is None:
        best_points = _regular_polygon(N)
    else:
        best_points = _normalize(best_points)

    return best_points
