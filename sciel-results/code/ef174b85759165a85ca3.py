"""Fifty equal spheres in a unit cube – maximise the common radius."""
import time
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist

N = 50
CUBE_HALF = 0.5
DEADLINE_MARGIN = 5.0  # seconds before the 150 s wall-clock limit


def _compute_radius(centres: np.ndarray) -> float:
    """Return the packing radius for a set of centres inside [-0.5,0.5]^3."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    wall = CUBE_HALF - np.max(np.abs(centres), axis=1)
    if np.min(wall) < 0.0:
        return 0.0
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d)
    return min(np.min(wall), pair_min / 2.0)


def _pair_dist_matrix(pts: np.ndarray) -> np.ndarray:
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    return d


def _lloyd_step(pts: np.ndarray) -> np.ndarray:
    """One Lloyd iteration: move each point to its Voronoi cell centroid."""
    try:
        vor = Voronoi(pts)
    except Exception:
        return pts
    new_pts = pts.copy()
    for i in range(len(pts)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        verts = vor.vertices[region]
        centroid = np.mean(verts, axis=0)
        new_pts[i] = centroid
    new_pts = np.clip(new_pts, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    if not np.all(np.isfinite(new_pts)):
        return pts
    return new_pts


def _lloyd_optimize(pts: np.ndarray, n_iter: int, deadline: float) -> np.ndarray:
    """Run Lloyd's algorithm to equalize Voronoi cell sizes."""
    best = pts.copy()
    best_score = _compute_radius(best)
    for it in range(n_iter):
        if time.time() > deadline:
            break
        new_pts = _lloyd_step(pts)
        if np.allclose(new_pts, pts, atol=1e-10):
            break
        s_new = _compute_radius(new_pts)
        if s_new >= best_score - 1e-12:
            pts = new_pts
            if s_new > best_score:
                best = pts.copy()
                best_score = s_new
        else:
            # Damped step
            for alpha in [0.5, 0.25, 0.1]:
                cand = pts + alpha * (new_pts - pts)
                cand = np.clip(cand, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                s_c = _compute_radius(cand)
                if s_c >= best_score - 1e-12:
                    pts = cand
                    if s_c > best_score:
                        best = cand.copy()
                        best_score = s_c
                    break
            else:
                break
    return best


def _softmin_radius(pts: np.ndarray, beta: float) -> float:
    """Smooth surrogate for the min radius using log-sum-exp."""
    wall = CUBE_HALF - np.max(np.abs(pts), axis=1)
    d = _pair_dist_matrix(pts)
    pair_half = d / 2.0
    # Combine wall clearances and pair distances
    all_vals = np.concatenate([wall, pair_half.ravel()])
    # Soft min: -log(sum(exp(-beta*x)))/beta
    min_val = np.min(all_vals)
    if min_val < -100:
        return min_val
    shifted = all_vals - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return min_val - log_sum


def _smooth_obj(pts: np.ndarray, beta: float) -> float:
    """Negative soft-min radius (to minimize)."""
    return -_softmin_radius(pts, beta)


def _smooth_polish(pts: np.ndarray, deadline: float, beta_start: float = 50.0,
                   beta_end: float = 200.0, n_iter: int = 300) -> np.ndarray:
    """Gradient-based polish using smooth surrogate with annealed beta."""
    best = pts.copy()
    best_score = _compute_radius(best)
    lr = 0.01
    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Anneal beta
        frac = it / max(n_iter - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        # Compute gradient numerically (or analytically via soft-min)
        # Use finite differences on the soft-min
        eps = 1e-5
        grad = np.zeros_like(pts)
        # For efficiency, compute gradient of soft-min analytically
        wall = CUBE_HALF - np.max(np.abs(pts), axis=1)
        d = _pair_dist_matrix(pts)
        pair_half = d / 2.0
        all_vals = np.concatenate([wall, pair_half.ravel()])
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        weights = np.exp(-beta * shifted)
        weights /= np.sum(weights)
        # Gradient of -log(sum(exp(-beta*x)))/beta w.r.t. x
        # d/dx_i [-log(sum_j w_j)/beta] where w_j = exp(-beta*(x_j - min))/sum
        # = (1/beta) * sum_j (w_j / sum_w) * beta * (delta_{j,i} - sum_k w_k/sum_w * delta_{k,i})
        # Simplified: grad_i = sum_j (w_j/sum_w) * (delta_{j,i} - avg_w)
        # But this is for the soft-min of the combined vector. Let's do it properly.
        # Actually, let's just use finite differences for the full objective.
        for i in range(N):
            for k in range(3):
                pts_p = pts.copy()
                pts_p[i, k] += eps
                pts_p = np.clip(pts_p, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                pts_m = pts.copy()
                pts_m[i, k] -= eps
                pts_m = np.clip(pts_m, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                f_p = _softmin_radius(pts_p, beta)
                f_m = _softmin_radius(pts_m, beta)
                grad[i, k] = (f_p - f_m) / (2 * eps)
        # Gradient ascent on soft-min radius
        pts_new = pts + lr * grad
        pts_new = np.clip(pts_new, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
        s_new = _compute_radius(pts_new)
        if s_new >= best_score - 1e-12:
            pts = pts_new
            if s_new > best_score:
                best = pts.copy()
                best_score = s_new
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.7
            if lr < 1e-6:
                break
    return best


def _repulsion_anneal(pts: np.ndarray, deadline: float, n_iter: int = 500) -> np.ndarray:
    """Annealed repulsion: push apart close pairs with decreasing threshold."""
    best = pts.copy()
    best_score = _compute_radius(best)
    lr = 0.02
    for it in range(n_iter):
        if time.time() > deadline:
            break
        frac = it / max(n_iter - 1, 1)
        threshold = 0.2 * (1.0 - 0.5 * frac)  # Decreasing threshold
        forces = np.zeros_like(pts)
        d = cdist(pts, pts)
        np.fill_diagonal(d, 1e10)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dist = d[i, j]
                if dist < threshold:
                    force_dir = (pts[i] - pts[j]) / max(dist, 1e-10)
                    forces[i] += force_dir * (threshold - dist)
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                if pts[i, k] > CUBE_HALF - 0.05:
                    forces[i, k] -= (pts[i, k] - (CUBE_HALF - 0.05)) * 2.0
                elif pts[i, k] < -CUBE_HALF + 0.05:
                    forces[i, k] += ((-CUBE_HALF + 0.05) - pts[i, k]) * 2.0
        pts_new = pts + lr * forces
        pts_new = np.clip(pts_new, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
        s_new = _compute_radius(pts_new)
        if s_new >= best_score - 1e-12:
            pts = pts_new
            if s_new > best_score:
                best = pts.copy()
                best_score = s_new
            lr = min(lr * 1.02, 0.05)
        else:
            lr *= 0.8
            if lr < 1e-6:
                break
    return best


def _local_search(pts: np.ndarray, deadline: float, n_restarts: int = 3) -> np.ndarray:
    """Final high-precision local search."""
    best = pts.copy()
    best_score = _compute_radius(best)
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        cur = best.copy()
        if restart > 0:
            cur = cur + np.random.normal(0, 0.005, size=cur.shape)
            cur = np.clip(cur, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
        lr = 0.01
        for it in range(1000):
            if time.time() > deadline:
                break
            # Compute forces: repulsion for pairs closer than current best radius * 1.5
            r = _compute_radius(cur)
            thresh = max(r * 1.5, 0.05)
            forces = np.zeros_like(cur)
            d = cdist(cur, cur)
            np.fill_diagonal(d, 1e10)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    dist = d[i, j]
                    if dist < thresh:
                        force_dir = (cur[i] - cur[j]) / max(dist, 1e-10)
                        forces[i] += force_dir * (thresh - dist) ** 2
            # Wall forces
            for i in range(N):
                for k in range(3):
                    wall_dist = CUBE_HALF - abs(cur[i, k])
                    if wall_dist < r:
                        if cur[i, k] > 0:
                            forces[i, k] -= (r - wall_dist) * 5.0
                        else:
                            forces[i, k] += (r - wall_dist) * 5.0
            cur_new = cur + lr * forces
            cur_new = np.clip(cur_new, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            s_new = _compute_radius(cur_new)
            if s_new >= best_score - 1e-12:
                cur = cur_new
                if s_new > best_score:
                    best = cur.copy()
                    best_score = s_new
                lr = min(lr * 1.03, 0.03)
            else:
                lr *= 0.7
                if lr < 1e-7:
                    break
    return best


def _seed_fcc(n: int) -> np.ndarray:
    """FCC lattice seed."""
    # FCC: simple cubic + face-centre offsets
    m = int(np.ceil(n ** (1.0 / 3.0)))
    pts = []
    a = 1.0 / m
    for ix in range(m + 1):
        for iy in range(m + 1):
            for iz in range(m + 1):
                x = -0.5 + ix * a
                y = -0.5 + iy * a
                z = -0.5 + iz * a
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    pts.append([x, y, z])
    # Face centres
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # (0.5, 0.5, 0) type faces
                pts.append([-0.5 + (ix + 0.5) * a, -0.5 + (iy + 0.5) * a, -0.5 + iz * a])
                pts.append([-0.5 + (ix + 0.5) * a, -0.5 + iy * a, -0.5 + (iz + 0.5) * a])
                pts.append([-0.5 + ix * a, -0.5 + (iy + 0.5) * a, -0.5 + (iz + 0.5) * a])
    pts = np.array(pts, dtype=np.float64)
    pts = pts[(pts >= -0.5 - 1e-10).all(axis=1) & (pts <= 0.5 + 1e-10).all(axis=1)]
    if len(pts) < n:
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(pts), 3))
        pts = np.vstack([pts, extra])
    else:
        idx = np.random.choice(len(pts), size=n, replace=False)
        pts = pts[idx]
    return pts[:n]


def _seed_grid(n: int) -> np.ndarray:
    """Simple cubic grid seed."""
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) < n:
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(grid), 3))
        grid = np.vstack([grid, extra])
    else:
        idx = np.random.choice(len(grid), size=n, replace=False)
        grid = grid[idx]
    return grid[:n]


def _seed_random(n: int) -> np.ndarray:
    return np.random.uniform(-0.45, 0.45, size=(n, 3))


def construct() -> np.ndarray:
    """Construct 50 sphere centres in the unit cube to maximise packing radius."""
    t_start = time.time()
    deadline = t_start + 150.0 - DEADLINE_MARGIN

    np.random.seed(42)

    # Generate multiple seeds
    seeds = [_seed_fcc(N), _seed_grid(N), _seed_random(N), _seed_random(N), _seed_random(N)]

    best = None
    best_score = 0.0

    for seed in seeds:
        if time.time() > deadline:
            break
        # Lloyd equalization
        pts = _lloyd_optimize(seed, n_iter=200, deadline=deadline)
        # Repulsion anneal
        pts = _repulsion_anneal(pts, deadline=deadline, n_iter=300)
        # Smooth polish
        pts = _smooth_polish(pts, deadline=deadline, n_iter=200)
        # Final local search
        pts = _local_search(pts, deadline=deadline, n_restarts=2)
        s = _compute_radius(pts)
        if s > best_score:
            best = pts.copy()
            best_score = s

    # Final polish on best
    if best is not None and time.time() < deadline:
        best = _smooth_polish(best, deadline=deadline, n_iter=200, beta_start=100, beta_end=500)
        best = _local_search(best, deadline=deadline, n_restarts=3)

    if best is None:
        best = _seed_grid(N)
    best = np.clip(best, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    return best
