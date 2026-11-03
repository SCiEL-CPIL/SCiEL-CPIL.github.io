import numpy as np
import time
import math
from scipy.optimize import least_squares, minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300

def _compute_radius_fast(pts):
    """Fast approximate radius for use inside optimization loops."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0.0:
        return 0.0
    # Use a subset-based estimate for speed during relaxation
    n = len(pts)
    if n > 50:
        idx = np.random.default_rng(42).choice(n, size=min(50, n), replace=False)
        sub = pts[idx]
        d = cdist(sub, sub)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
    else:
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _compute_radius(pts):
    """Compute the packing radius for given points (max feasible common radius)."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0.0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum

def _hex_init(n_points, seed=0):
    """Multi-seed, multi-rotation hexagonal lattice initializer.

    Tries several spacings and rotations, keeps the best-scoring selection.
    """
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    for attempt in range(14):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]].copy()

        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts

def _force_relax(pts, n_steps=1500, deadline=None):
    """Force-directed relaxation with Hertzian overlap^4 repulsion + wall force."""
    pos = np.asarray(pts, dtype=np.float64).copy()
    best_pos = pos.copy()
    best_r = _compute_radius_fast(best_pos)
    n = len(pos)

    r_est = best_r
    lr = 0.02
    cooldown = 0.998
    rng = np.random.default_rng(12345)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)

        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        target = 2.0 * r_est
        mask = dists < target
        safe_dists = np.where(mask, dists, 1.0)

        excess = np.where(mask, target - dists, 0.0)
        force_mag = np.where(mask, 4.0 * excess**3 / safe_dists, 0.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)

        # Wall force: push inward when too close to boundary
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.03
        wall_excess = np.clip(norms - (1.0 - wall_margin), 0.0, None)
        wall_mag = 50.0 * wall_excess**3
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        forces += wall_mag[:, np.newaxis] * (-pos / safe_norms[:, np.newaxis])

        # Integrate with adaptive step
        pos = pos + lr * forces
        # Project inside container
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] /= norms[mask_out, np.newaxis]

        r_now = _compute_radius_fast(pos)
        if r_now > best_r:
            best_r = r_now
            best_pos = pos.copy()
            r_est = best_r
        else:
            r_est = 0.98 * r_est + 0.02 * best_r

        lr *= cooldown

    return best_pos

def _soft_min_objective_and_grad(x, T, edge_indices=None):
    """Minimize -softmin(constraints). Returns (obj, grad.ravel())."""
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    if edge_indices is None:
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
    else:
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])

    all_vals = np.concatenate([pair_half, boundary])
    obj = -_soft_min_logsumexp(all_vals, T)

    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights

    grad_pts = np.zeros_like(pts)

    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary

    if edge_indices is None:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        W = np.zeros((n, n))
        n_pairs = len(pair_half)
        iu = np.triu_indices(n, k=1)
        W[iu] = grad_all[:n_pairs]
        W[iu[1], iu[0]] = grad_all[:n_pairs]
        weighted_unit = W[:, :, np.newaxis] * unit_vecs
        grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
        grad_pts += grad_pairs
    else:
        for k, (i, j) in enumerate(edge_indices):
            w = grad_all[k]
            if w == 0.0:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            unit_vec = diff / d
            grad_pts[i] += 0.5 * w * unit_vec
            grad_pts[j] -= 0.5 * w * unit_vec

    return obj, grad_pts.ravel()

def _active_set_edges(pts, target_r):
    """Delaunay edges plus any pair closer than ~2.2*target_r (bottleneck pairs)."""
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    if i == j:
                        continue
                    edges.add((min(i, j), max(i, j)))
    except Exception:
        edges = set()

    # Add near pairs
    d = cdist(pts, pts)
    iu = np.triu_indices(len(pts), k=1)
    near = d[iu] < 2.2 * target_r
    for idx in np.where(near)[0]:
        i, j = iu[0][idx], iu[1][idx]
        edges.add((min(i, j), max(i, j)))

    return list(edges)

def _polish_active(pts, T_start=0.02, T_end=0.001, n_iter=6, deadline=None):
    """Sparse active-set L-BFGS-B polish with progressively tighter temperatures."""
    cur = pts.copy()
    cur_r = _compute_radius(cur)
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        frac = it / max(1, n_iter - 1)
        T = T_start * (T_end / T_start) ** frac
        edges = _active_set_edges(cur, max(cur_r, 1e-6))

        def obj(x):
            return _soft_min_objective_and_grad(x, T, edges)[0]

        def grad(x):
            return _soft_min_objective_and_grad(x, T, edges)[1]

        try:
            res = minimize(
                obj, cur.flatten(), jac=grad, method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * len(cur)),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            cand = res.x.reshape(-1, 2)
            norms = np.linalg.norm(cand, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                cand = cand.copy()
                cand[mask] /= norms[mask, np.newaxis]
            cand_r = _compute_radius(cand)
            if cand_r > cur_r:
                cur = cand
                cur_r = cand_r
        except Exception:
            pass
    return cur, cur_r

def _polish_full(pts, T=0.01, deadline=None):
    """Full-pairwise soft-min polish via L-BFGS-B with analytic gradient."""
    cur = pts.copy()
    cur_r = _compute_radius(cur)
    try:
        res = minimize(
            lambda x: _soft_min_objective_and_grad(x, T, None)[0],
            cur.flatten(),
            jac=lambda x: _soft_min_objective_and_grad(x, T, None)[1],
            method='L-BFGS-B',
            bounds=[(-1.0, 1.0)] * (2 * len(cur)),
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        cand = res.x.reshape(-1, 2)
        norms = np.linalg.norm(cand, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            cand = cand.copy()
            cand[mask] /= norms[mask, np.newaxis]
        cand_r = _compute_radius(cand)
        if cand_r > cur_r:
            cur = cand
            cur_r = cand_r
    except Exception:
        pass
    return cur, cur_r

def construct():
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill

    def remaining():
        return time_limit - (time.time() - start_time)

    # Multi-seed hexagonal starts
    n_starts = 8
    starts = []
    for s in range(n_starts):
        if remaining() < 20.0:
            break
        p = _hex_init(N, seed=1000 + s)
        starts.append(p)

    best_points = starts[0].copy()
    best_score = _compute_radius(best_points)

    # Phase 1: force-relax each start, then active-set polish
    for p in starts:
        if remaining() < 15.0:
            break
        relaxed = _force_relax(p, n_steps=1200, deadline=start_time + time_limit - 12.0)
        r1 = _compute_radius(relaxed)
        if r1 > best_score:
            best_score = r1
            best_points = relaxed.copy()

        # active-set polish
        if remaining() > 10.0:
            polished, rp = _polish_active(relaxed, T_start=0.03, T_end=0.002, n_iter=5,
                                           deadline=start_time + time_limit - 8.0)
            if rp > best_score:
                best_score = rp
                best_points = polished.copy()

    # Phase 2: Skip full-pairwise (too slow for N=300), use more active-set passes
    if remaining() > 8.0:
        polished, rp = _polish_active(best_points, T_start=0.015, T_end=0.001, n_iter=6,
                                       deadline=start_time + time_limit - 5.0)
        if rp > best_score:
            best_score = rp
            best_points = polished.copy()

    # Phase 3: final active-set passes with tighter temps for final squeeze
    for it in range(4):
        if remaining() < 6.0:
            break
        polished, rp = _polish_active(best_points, T_start=0.006, T_end=0.0003, n_iter=5,
                                      deadline=start_time + time_limit - 4.0)
        if rp > best_score:
            best_score = rp
            best_points = polished.copy()

    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        return np.zeros((N, 2))

    return best_points
