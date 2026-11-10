# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


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


def _regular_polygon_config(n: int, center: tuple, radius: float = 0.4) -> np.ndarray:
    """Regular n-gon centered at center with given radius."""
    cx, cy = center
    points = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        points.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])
    return np.array(points)


def _local_refine(points: np.ndarray, deadline: float, max_iters: int = 50000) -> np.ndarray:
    """Random-walk local search on the true score, targeting the worst triangle."""
    best = points.copy()
    best_score = _compute_score(best)
    n = len(best)
    tri = _get_triangle_indices(n)

    step_size = 0.015
    min_step = 1e-9
    decay = 0.90
    stall = 0
    max_stall = 25

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
            for _ in range(12):
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
                max_stall = max(12, max_stall // 2)
    return best


def _nelder_mead(points: np.ndarray, deadline: float, max_iters: int = 20000,
                 initial_step: float = 0.02) -> np.ndarray:
    """
    Nelder-Mead style simplex optimization on the point set.
    We flatten the 2D points into a 26-d vector and optimize.
    """
    n = points.shape[0]
    dim = 2 * n
    x = points.flatten().copy()
    
    def objective(flat: np.ndarray) -> float:
        pts = flat.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return -1e9
        return _compute_score(pts)
    
    # Build initial simplex
    simplex = [x.copy()]
    for i in range(dim):
        pert = x.copy()
        pert[i] += initial_step
        simplex.append(pert)
    
    # Evaluate
    values = [objective(s) for s in simplex]
    
    best_idx = np.argmax(values)
    best_x = simplex[best_idx].copy()
    best_val = values[best_idx]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Sort simplex by value (worst last)
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Update best
        if values[-1] > best_val:
            best_val = values[-1]
            best_x = simplex[-1].copy()
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr >= values[0]:
            # Try expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe > fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr >= values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            if fr < values[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc >= fr:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
                    break
            else:
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc >= values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
                    break
    
    return best_x.reshape(n, 2)


def _simulated_annealing(points: np.ndarray, deadline: float,
                         T_init: float = 0.02, T_min: float = 1e-9,
                         cooling_rate: float = 0.9993, max_iters: int = 40000) -> np.ndarray:
    best = points.copy()
    cur = points.copy()
    cur_score = _compute_score(cur)
    best_score = cur_score
    n = len(cur)

    step_size = 0.025
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
        if (accept + reject) % 200 == 0 and (accept + reject) > 0:
            rate = accept / (accept + reject)
            if rate < 0.12:
                step_size *= 0.75
            elif rate > 0.80:
                step_size *= 1.2
            step_size = max(min_step, min(step_size, 0.06))
            accept = 0
            reject = 0
    return best


def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # leave 40s margin before 630s kill

    best_points = None
    best_score = 0.0

    # Build diverse initial configurations
    configs = []
    # Hex configurations with various radii
    for r1 in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25]:
        for r2 in [0.30, 0.35, 0.40, 0.42, 0.45, 0.48]:
            configs.append(_build_hex_configuration(r1, r2, (0.5, 0.5)))
    # Concentric
    configs.append(_build_concentric_config(13, (0.5, 0.5)))
    # Grid
    configs.append(_build_grid_config(13))
    # Regular polygons
    configs.append(_regular_polygon_config(13, (0.5, 0.5), 0.40))
    configs.append(_regular_polygon_config(13, (0.5, 0.5), 0.45))
    configs.append(_regular_polygon_config(13, (0.5, 0.5), 0.35))
    # Random
    for seed in range(8):
        configs.append(_build_random_config(13, seed))

    # Score all configs
    scored = []
    for cfg in configs:
        if time.time() > deadline - 450:
            break
        s = _compute_score(cfg)
        scored.append((s, cfg))
    scored.sort(key=lambda x: -x[0])

    top_configs = [cfg for _, cfg in scored[:6]]

    # Phase 1: Simulated annealing on top configs
    sa_deadline = deadline - 250
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline,
                                         T_init=0.02, T_min=1e-9,
                                         cooling_rate=0.9993, max_iters=35000)
        s = _compute_score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()

    # Phase 2: Nelder-Mead refinement
    if best_points is not None:
        nm_deadline = deadline - 150
        nm_result = _nelder_mead(best_points, nm_deadline, max_iters=15000,
                                  initial_step=0.015)
        s = _compute_score(nm_result)
        if s > best_score:
            best_score = s
            best_points = nm_result.copy()

    # Phase 3: Local search refinement
    if best_points is not None:
        ls_deadline = deadline - 60
        ls_result = _local_refine(best_points, ls_deadline, max_iters=60000)
        s = _compute_score(ls_result)
        if s > best_score:
            best_score = s
            best_points = ls_result.copy()

    # Phase 4: Random restarts with quick refinement
    if best_points is not None and time.time() < deadline - 80:
        for _ in range(8):
            if time.time() > deadline - 50:
                break
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.04
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            quick = _local_refine(perturbed, deadline - 50, max_iters=12000)
            s = _compute_score(quick)
            if s > best_score:
                best_score = s
                best_points = quick.copy()

    # Safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_hex_configuration(0.18, 0.42, (0.5, 0.5))

    return best_points


# EVOLVE-BLOCK-END
