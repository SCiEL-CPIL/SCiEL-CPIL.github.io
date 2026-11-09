import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, differential_evolution
import sys

N = 20


def _vertex_score(distances):
    d = np.sort(distances)
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    k = int(np.argmin(diffs))
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)


def _is_valid_convex(points):
    n = len(points)
    # repeated points
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    if np.min(dmat) < 1e-6:
        return False
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-8):
        return False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    return True


def _polygon_score(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_c = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_c)
    if mean_dist <= 0.01:
        return 0.0
    points = points / mean_dist
    if not _is_valid_convex(points):
        return 0.0
    min_score = 1.0
    # pairwise distances matrix
    dm = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(dm, 0.0)
    for i in range(N):
        s = _vertex_score(dm[i])
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            break
    return min_score


def _points_to_params(points):
    centroid = np.mean(points, axis=0)
    pc = points - centroid
    radii = np.linalg.norm(pc, axis=1)
    angles = np.arctan2(pc[:, 1], pc[:, 0])
    angles = np.mod(angles, 2 * np.pi)
    idx = np.argsort(angles)
    radii = radii[idx]
    angles = angles[idx]
    base = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    offsets = angles - base
    offsets = np.where(offsets > np.pi, offsets - 2 * np.pi, offsets)
    offsets = np.where(offsets < -np.pi, offsets + 2 * np.pi, offsets)
    return np.concatenate([offsets, radii])


def _params_to_points(params):
    n = N
    offs = params[:n]
    radii = np.maximum(params[n:], 0.05)
    base = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base + offs
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _objective(params):
    pts = _params_to_points(params)
    sc = _polygon_score(pts)
    if sc <= 0.0:
        return 1e6
    return -sc


def _objective_softmin(params):
    """Smooth surrogate: uses a soft-min over vertex scores to make objective smoother."""
    pts = _params_to_points(params)
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists_to_c = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_c)
    if mean_dist <= 0.01:
        return 1e6
    pts = pts / mean_dist
    if not _is_valid_convex(pts):
        return 1e6
    dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dm, 0.0)
    scores = np.array([_vertex_score(dm[i]) for i in range(N)])
    # soft-min with temperature
    T = 0.05
    log_scores = np.log(np.maximum(scores, 1e-10))
    m = np.min(log_scores)
    z = np.exp((log_scores - m) / T)
    soft_min = m * T + T * np.log(np.sum(z))
    # convert back to score scale
    soft_min_score = np.exp(soft_min)
    if soft_min_score <= 0.0:
        return 1e6
    return -soft_min_score


def _local_refine(points, max_iter=200, step=0.005, deadline=None):
    """Coordinate-descent local refinement with adaptive step."""
    n = N
    cur = points.copy()
    cur_score = _polygon_score(cur)
    s = step
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try moving each vertex in 4 directions with current step
        for i in range(n):
            for dx in (-s, s):
                for dy in (-s, s):
                    trial = cur.copy()
                    trial[i] += [dx, dy]
                    ts = _polygon_score(trial)
                    if ts > cur_score:
                        cur = trial
                        cur_score = ts
                        improved = True
        if improved:
            s *= 0.95
        else:
            s *= 0.8
            if s < 1e-5:
                break
    return cur, cur_score


def _nm_refine(params, deadline, max_iter=3000, maxfev=6000):
    """Nelder-Mead refinement on parameter space."""
    try:
        res = minimize(
            _objective,
            params,
            method='Nelder-Mead',
            options={'xatol': 1e-7, 'fatol': 1e-8, 'maxiter': max_iter, 'maxfev': maxfev, 'adaptive': True}
        )
        if res.fun < 1e5:
            sc = -res.fun
            pts = _params_to_points(res.x)
            return res.x, sc, pts
    except Exception:
        pass
    return params, None, None


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    n = N

    # Baseline: regular polygon
    best_points = _params_to_points(np.zeros(2 * n))
    best_score = _polygon_score(best_points)
    best_params = np.zeros(2 * n)

    # Phase 1: Differential evolution on general (angle-offset, radius) parameterization
    # Angle offsets bounded to a fraction of the mean angular spacing
    ang_step = 2.0 * np.pi / n
    angle_bound = 0.35 * ang_step
    radius_bounds = (0.6, 1.4)
    bounds = [(-angle_bound, angle_bound)] * n + [radius_bounds] * n

    # Run DE with multiple seeds and increasing popsize
    seeds = [42, 123, 456, 789, 2024]
    for seed in seeds:
        if time.time() > deadline - 180.0:
            break
        try:
            res = differential_evolution(
                _objective,
                bounds,
                maxiter=25,
                popsize=10,
                tol=1e-5,
                seed=seed,
                polish=False,
                mutation=(0.3, 0.9),
                recombination=0.7
            )
            if res.fun < 1e5:
                sc = -res.fun
                if sc > best_score:
                    best_score = sc
                    best_points = _params_to_points(res.x)
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement from best DE result
    if time.time() < deadline - 120.0:
        p, sc, pts = _nm_refine(best_params, deadline, max_iter=4000, maxfev=8000)
        if sc is not None and sc > best_score:
            best_score = sc
            best_points = pts
            best_params = p

    # Phase 3: Additional NM restarts with small perturbations
    for restart in range(5):
        if time.time() > deadline - 80.0:
            break
        perturbed = best_params + 0.005 * np.random.randn(2 * n)
        p, sc, pts = _nm_refine(perturbed, deadline, max_iter=2000, maxfev=4000)
        if sc is not None and sc > best_score:
            best_score = sc
            best_points = pts
            best_params = p

    # Phase 4: Soft-min objective NM for smoother landscape
    if time.time() < deadline - 60.0:
        try:
            res = minimize(
                _objective_softmin,
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-8, 'maxiter': 3000, 'maxfev': 6000, 'adaptive': True}
            )
            if res.fun < 1e5:
                sc = -res.fun
                pts = _params_to_points(res.x)
                # Verify with hard score
                hard_sc = _polygon_score(pts)
                if hard_sc > best_score:
                    best_score = hard_sc
                    best_points = pts
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Final coordinate-descent polish in Cartesian space
    if time.time() < deadline - 20.0:
        polished, polished_score = _local_refine(best_points, max_iter=150, step=0.003, deadline=deadline - 5.0)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished

    return best_points


if __name__ == "__main__":
    t0 = time.time()
    pts = equidistant_points_polygon20()
    print("Score:", _polygon_score(pts))
    print("Time: %.1f s" % (time.time() - t0))
