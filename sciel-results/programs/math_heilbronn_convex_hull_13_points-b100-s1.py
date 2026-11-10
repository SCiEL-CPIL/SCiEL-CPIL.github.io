# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _get_triangle_indices(n: int):
    """Precompute triangle index arrays for n points."""
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull (true, nonsmooth)."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 0:
        return 0.0

    i_idx, j_idx, k_idx = _get_triangle_indices(n)
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    min_area = float(np.min(0.5 * cross))
    if min_area <= 0:
        return 0.0
    return min_area / A_hull


def _softmin_value(points: np.ndarray, tri: tuple, alpha: float) -> float:
    """Soft-min of triangle areas (smooth surrogate)."""
    pi = points[tri[0]]
    pj = points[tri[1]]
    pk = points[tri[2]]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    m = np.min(areas)
    if m <= 0:
        return -1e9
    return m - np.log(np.sum(np.exp(-alpha * (areas - m)))) / alpha


def _softmin_grad(points: np.ndarray, tri: tuple, alpha: float):
    """
    Gradient of the soft-min objective w.r.t. each point coordinate.
    Returns (grad, value) where grad has shape (n, 2).
    """
    n = points.shape[0]
    i_idx, j_idx, k_idx = tri
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    m = np.min(areas)
    if m <= 0:
        return np.zeros((n, 2)), -1e9
    weights = np.exp(-alpha * (areas - m))
    weights /= np.sum(weights)
    val = m - np.log(np.sum(weights)) / alpha

    # Per-triangle gradient of area = 0.5 * |v1 x v2|
    raw = v1x * v2y - v1y * v2x
    sign = np.sign(raw)
    sign[sign == 0] = 1.0

    darea_dpi = np.column_stack([
        -0.5 * sign * v2y + 0.5 * sign * v1y,
        0.5 * sign * v2x - 0.5 * sign * v1x
    ])
    darea_dpj = np.column_stack([
        0.5 * sign * v2y,
        -0.5 * sign * v2x
    ])
    darea_dpk = np.column_stack([
        -0.5 * sign * v1y,
        0.5 * sign * v1x
    ])

    grad = np.zeros((n, 2))
    np.add.at(grad, i_idx, weights[:, None] * darea_dpi)
    np.add.at(grad, j_idx, weights[:, None] * darea_dpj)
    np.add.at(grad, k_idx, weights[:, None] * darea_dpk)
    return grad, val


def _softmin_optimize(points: np.ndarray, deadline: float, alpha: float = 40.0) -> np.ndarray:
    """
    L-BFGS-B optimization on the negative soft-min objective.
    Returns the best true-score point set found.
    """
    best = points.copy()
    best_score = _compute_score(best)
    n = len(best)
    tri = _get_triangle_indices(n)
    
    def objective_and_grad(flat_points):
        pts = flat_points.reshape(n, 2)
        grad, val = _softmin_grad(pts, tri, alpha)
        # We want to maximize val, so minimize -val
        # If val is -1e9 (degenerate), return a large penalty
        if val < -1e8:
            return 1e9, np.zeros_like(flat_points)
        return -val, -grad.reshape(-1)

    x0 = points.reshape(-1)
    bounds = [(0.0, 1.0)] * (n * 2)
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < 0: # Check if it found something valid
            new_pts = res.x.reshape(n, 2)
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best = new_pts
                best_score = new_score
    except Exception:
        pass
        
    return best


def _build_hex_configuration(r1: float, r2: float, center: tuple) -> np.ndarray:
    """1 center + 6 on inner hexagon + 6 on outer hexagon = 13 points."""
    cx, cy = center
    points = [[cx, cy]]
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    return np.array(points)


def _build_concentric_config(n: int, center: tuple) -> np.ndarray:
    cx, cy = center
    points = [[cx, cy]]
    r1 = 0.18
    for i in range(5):
        angle = 2 * np.pi * i / 5
        points.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    r2 = 0.38
    for i in range(7):
        angle = 2 * np.pi * i / 7
        points.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    return np.array(points[:n])


def _build_grid_config(n: int) -> np.ndarray:
    points = []
    for i in range(4):
        for j in range(4):
            if len(points) >= n:
                break
            points.append([0.1 + i * 0.26, 0.1 + j * 0.26])
    return np.array(points[:n])


def _build_random_config(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 0.8 + 0.1


def _local_refine(points: np.ndarray, deadline: float, max_iters: int = 50000) -> np.ndarray:
    """Random-walk local search on the true score, targeting the worst triangle."""
    best = points.copy()
    best_score = _compute_score(best)
    n = len(best)
    tri = _get_triangle_indices(n)

    step_size = 0.02
    min_step = 1e-9
    decay = 0.92
    stall = 0
    max_stall = 30

    for it in range(max_iters):
        if time.time() > deadline:
            break
        # worst triangle
        pi = best[tri[0]]
        pj = best[tri[1]]
        pk = best[tri[2]]
        v1x = pj[:, 0] - pi[:, 0]
        v1y = pj[:, 1] - pi[:, 1]
        v2x = pk[:, 0] - pi[:, 0]
        v2y = pk[:, 1] - pi[:, 1]
        cross = np.abs(v1x * v2y - v1y * v2x)
        min_idx = np.argmin(0.5 * cross)
        wi, wj, wk = tri[0][min_idx], tri[1][min_idx], tri[2][min_idx]

        improved = False
        for idx in (wi, wj, wk):
            for _ in range(8):
                dx = np.random.randn() * step_size
                dy = np.random.randn() * step_size
                cand = best.copy()
                cand[idx, 0] = np.clip(best[idx, 0] + dx, 0, 1)
                cand[idx, 1] = np.clip(best[idx, 1] + dy, 0, 1)
                s = _compute_score(cand)
                if s > best_score:
                    best = cand
                    best_score = s
                    improved = True
                    stall = 0
                    break
            if improved:
                break
        if not improved:
            stall += 1
            if stall >= max_stall:
                step_size *= decay
                if step_size < min_step:
                    break
                stall = 0
                max_stall = max(15, max_stall // 2)
    return best


def _simulated_annealing(points: np.ndarray, deadline: float,
                         T_init: float = 0.03, T_min: float = 1e-9,
                         cooling_rate: float = 0.999, max_iters: int = 40000) -> np.ndarray:
    best = points.copy()
    cur = points.copy()
    cur_score = _compute_score(cur)
    best_score = cur_score
    n = len(cur)

    step_size = 0.03
    min_step = 1e-9
    accept = 0
    reject = 0

    for it in range(max_iters):
        if time.time() > deadline:
            break
        T = max(T_init * (cooling_rate ** it), T_min)
        idx = np.random.randint(n)
        dx = np.random.randn() * step_size
        dy = np.random.randn() * step_size
        cand = cur.copy()
        cand[idx, 0] = np.clip(cur[idx, 0] + dx, 0, 1)
        cand[idx, 1] = np.clip(cur[idx, 1] + dy, 0, 1)
        cand_score = _compute_score(cand)
        if cand_score >= cur_score:
            cur = cand
            cur_score = cand_score
            accept += 1
            if cur_score > best_score:
                best_score = cur_score
                best = cand.copy()
        else:
            delta = cur_score - cand_score
            if np.random.random() < np.exp(-delta / max(T, 1e-12)):
                cur = cand
                cur_score = cand_score
                accept += 1
            else:
                reject += 1
        if (accept + reject) % 300 == 0 and (accept + reject) > 0:
            rate = accept / (accept + reject)
            if rate < 0.15:
                step_size *= 0.8
            elif rate > 0.85:
                step_size *= 1.15
            step_size = max(min_step, min(step_size, 0.08))
            accept = 0
            reject = 0
    return best


def _nelder_mead_polish(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    High-precision polish using Nelder-Mead on the negative true score.
    """
    n = points.shape[0]
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        # Penalize points outside [0,1] to keep them in bounds
        # Nelder-Mead doesn't handle bounds well, so we rely on the landscape
        # being flat or penalizing outside.
        # However, _compute_score returns 0 for invalid, which is a minimum.
        # We want to maximize score, so minimize -score.
        s = _compute_score(pts)
        if s == 0.0:
            return 1e9
        return -s

    x0 = points.reshape(-1)
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            new_pts = res.x.reshape(n, 2)
            # Ensure points are still valid and in bounds
            if np.all(new_pts >= 0) and np.all(new_pts <= 1) and np.all(np.isfinite(new_pts)):
                new_score = _compute_score(new_pts)
                cur_score = _compute_score(points)
                if new_score > cur_score:
                    return new_pts
    except Exception:
        pass
    return points


def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 600  # evaluator kills at 630s; leave 30s margin

    best_points = None
    best_score = 0.0

    configs = []
    for r1 in [0.15, 0.18, 0.20, 0.22]:
        for r2 in [0.35, 0.40, 0.42, 0.45]:
            configs.append(_build_hex_configuration(r1, r2, (0.5, 0.5)))
    configs.append(_build_concentric_config(13, (0.5, 0.5)))
    configs.append(_build_grid_config(13))
    for seed in range(10):
        configs.append(_build_random_config(13, seed))

    scored = []
    for cfg in configs:
        if time.time() > deadline - 400:
            break
        s = _compute_score(cfg)
        scored.append((s, cfg))
    scored.sort(key=lambda x: -x[0])

    top_configs = [cfg for _, cfg in scored[:5]]

    # Phase 1: simulated annealing on top configs
    sa_deadline = deadline - 200
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline,
                                         T_init=0.025, T_min=1e-9,
                                         cooling_rate=0.9992, max_iters=30000)
        s = _compute_score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()

    # Phase 2: smooth soft-min gradient ascent on the best point
    if best_points is not None:
        gm_deadline = deadline - 120
        for alpha in [30.0, 60.0]:
            if time.time() > gm_deadline:
                break
            gm_result = _softmin_optimize(best_points, gm_deadline, alpha=alpha)
            s = _compute_score(gm_result)
            if s > best_score:
                best_score = s
                best_points = gm_result.copy()

    # Phase 3: local search refinement
    if best_points is not None:
        ls_deadline = deadline - 50
        ls_result = _local_refine(best_points, ls_deadline, max_iters=50000)
        s = _compute_score(ls_result)
        if s > best_score:
            best_score = s
            best_points = ls_result.copy()

    # Phase 4: Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 30:
        nm_result = _nelder_mead_polish(best_points, deadline - 10)
        s = _compute_score(nm_result)
        if s > best_score:
            best_score = s
            best_points = nm_result.copy()

    # Safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_hex_configuration(0.2, 0.42, (0.5, 0.5))

    return best_points


# EVOLVE-BLOCK-END
