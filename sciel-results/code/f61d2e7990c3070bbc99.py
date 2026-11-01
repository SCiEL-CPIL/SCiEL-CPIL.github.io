"""
Refined packing of 50 equal circles in a unit circle.
Strategy:
1. Clean, numerically-stable soft-min surrogate (shifted log-sum-exp) with analytic gradient.
2. Multi-stage beta annealing (coarse to fine) to escape local minima.
3. Force-directed pre-conditioning for fast initial improvement.
4. Local search with adaptive step size for fine-tuning.
5. Diverse initializations (ring, hex lattice, random) to cover symmetry classes.
6. Strict time management with explicit deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 145.0  # 5s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0

    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists_origin > 1.0 + 1e-12):
        return 0.0

    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < -1e-12:
        return 0.0

    if N < 2:
        return max(0.0, r_cont)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists)

    return max(0.0, min(r_cont, r_pair))


def _ring_init():
    """
    Generate ring-based initialization matching known optimal structure:
    1 (center) + 6 + 12 + 18 + 13 = 50
    """
    rings = [
        (0.0, 1),
        (0.2516, 6),
        (0.5032, 12),
        (0.7548, 18),
        (0.945, 13),
    ]

    pts = []
    for ring_idx, (radius, count) in enumerate(rings):
        if count == 1:
            pts.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            offset = ring_idx * (np.pi / 6.0)
            for a in angles:
                a += offset
                pts.append([radius * np.cos(a), radius * np.sin(a)])

    pts = np.array(pts[:N])
    return pts


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0

    for spacing in np.linspace(0.22, 0.30, 40):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    if best_pts is not None:
        return best_pts

    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_init(seed=0):
    """Random initialization in unit disk."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _jittered_ring_init():
    """Ring init with small random perturbations to break symmetry."""
    base = _ring_init()
    rng = np.random.RandomState(7)
    jitter = rng.uniform(-0.02, 0.02, base.shape)
    pts = base + jitter
    radii = np.hypot(pts[:, 0], pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / radii[mask, np.newaxis]
    return pts


def _soft_min_obj_and_grad(x, beta):
    """
    Clean soft-min objective with shifted log-sum-exp for numerical stability.
    Minimizes -soft_min(min_pair_dist/2, 1 - max_dist_to_origin).
    
    soft_min(vals) = m - (1/beta) * log(sum(exp(-beta * (vals - m))))
    where m = min(vals). This is numerically stable since vals - m >= 0.
    """
    pts = x.reshape(N, 2)

    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]

    # Boundary distances
    norms = np.hypot(pts[:, 0], pts[:, 1])
    bound_dists = 1.0 - norms

    # Combine all constraint values we want to maximize
    vals = np.concatenate([pair_dists / 2.0, bound_dists])

    # Stable soft-min
    m = np.min(vals)
    shifted = vals - m  # >= 0
    exp_terms = np.exp(-beta * shifted)  # <= 1, safe
    S = np.sum(exp_terms)

    soft_min_val = m - (1.0 / beta) * np.log(S)
    obj_val = -soft_min_val

    # Gradient: grad_obj = -grad_soft_min
    # grad_soft_min = sum_k w_k * grad(x_k) where w_k = exp_terms[k] / S
    weights = exp_terms / S

    grad_pts = np.zeros_like(pts)

    # Pair contributions
    pair_weights = weights[:N * (N - 1) // 2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (pts[i] - pts[j]) / d
            # grad_obj[c_i] += -w * 0.5 * vec
            # grad_obj[c_j] += +w * 0.5 * vec
            grad_pts[i] -= w * 0.5 * vec
            grad_pts[j] += w * 0.5 * vec

    # Boundary contributions
    bound_weights = weights[N * (N - 1) // 2:]
    for i in range(N):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm > 1e-8:
            vec = pts[i] / norm
            # grad_obj[c_i] += -w * (-vec) = +w * vec
            grad_pts[i] += w * vec

    # Penalty for points outside the unit circle
    outside = norms > 1.0
    if np.any(outside):
        pen = 5.0 * np.sum(np.maximum(norms - 1.0, 0.0) ** 2)
        obj_val += pen
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        grad_pen = 10.0 * np.maximum(norms - 1.0, 0.0)[:, np.newaxis] * (pts / safe_norms[:, np.newaxis])
        grad_pts += grad_pen

    return obj_val, grad_pts.flatten()


def _optimize_from_start(pts0, time_limit, betas=None):
    """
    Progressive soft-min optimization with L-BFGS-B using beta annealing.
    """
    if betas is None:
        betas = [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()

    for beta in betas:
        if time.time() > time_limit:
            break

        def obj(x):
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val

        def grad(x):
            _, g = _soft_min_obj_and_grad(x, beta)
            return g

        try:
            remaining = max(1.0, time_limit - time.time())
            maxiter = max(50, min(500, int(remaining * 10)))

            res = minimize(
                obj,
                grad,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            new_pts = res.x.reshape(N, 2)

            # Project back to unit disk
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]

            new_score = _score(new_pts)

            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
        except Exception:
            pass

    return best_pts


def _local_search(pts0, time_limit):
    """
    Local search: try small perturbations and keep improvements.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(12345)

    step_size = 0.05
    no_improve = 0

    while time.time() < time_limit and no_improve < 300:
        perturb = rng.uniform(-step_size, step_size, pts.shape)
        new_pts = pts + perturb

        radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]

        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve = 0
            step_size = min(0.1, step_size * 1.05)
        else:
            no_improve += 1
            step_size *= 0.95

    return best_pts


def _force_directed(pts0, time_limit, iters=2000):
    """
    Force-directed layout for rapid improvement.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()

    lr = 0.003
    decay = 0.999

    for it in range(iters):
        if time.time() > time_limit:
            break

        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)

        r_est = 0.5 * np.min(dists[dists < np.inf])
        if r_est < 1e-6:
            r_est = 0.05

        forces = np.zeros_like(pts)

        close_mask = (dists < 3.0 * r_est) & (dists > 1e-10)
        if np.any(close_mask):
            safe_dists = np.where(close_mask, dists, 1.0)
            inv_d3 = np.where(close_mask, 1.0 / (safe_dists ** 3), 0.0)
            np.fill_diagonal(inv_d3, 0.0)

            row_sums = np.sum(inv_d3, axis=1, keepdims=True)
            forces += pts * row_sums - inv_d3 @ pts

        dists_origin = np.hypot(pts[:, 0], pts[:, 1])
        margin = 1.0 - dists_origin
        safe_orig = np.where(dists_origin > 1e-10, dists_origin, 1.0)
        boundary_force_mag = np.where(margin < r_est, 2.0 * (r_est - margin), 0.0)
        forces -= boundary_force_mag[:, np.newaxis] * (pts / safe_orig[:, np.newaxis])

        forces -= 0.0005 * pts

        pts += lr * forces

        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis]

        pts = np.nan_to_num(pts, nan=0.0, posinf=0.9, neginf=-0.9)

        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

        lr *= decay

    return best_pts


def construct():
    start_time = time.time()

    # 1. Generate multiple initializations
    initializations = [
        _ring_init(),
        _jittered_ring_init(),
        _hex_lattice_init(),
        _random_init(42),
        _random_init(123),
        _random_init(7),
    ]

    init_scores = [_score(p) for p in initializations]
    best_init_idx = np.argmax(init_scores)
    best_pts = initializations[best_init_idx].copy()
    best_score = init_scores[best_init_idx]

    # 2. Force-directed from best init (quick improvement)
    fd_deadline = _DEADLINE - 95.0
    if time.time() < fd_deadline:
        try:
            fd_pts = _force_directed(best_pts, fd_deadline, iters=2000)
            fd_score = _score(fd_pts)
            if fd_score > best_score:
                best_pts = fd_pts
                best_score = fd_score
        except Exception:
            pass

    # 3. Try force-directed from other good inits
    if time.time() < _DEADLINE - 85.0:
        for i, init in enumerate(initializations):
            if i == best_init_idx:
                continue
            if init_scores[i] < best_score * 0.92:
                continue
            if time.time() > _DEADLINE - 75.0:
                break
            try:
                fd_pts = _force_directed(init, _DEADLINE - 75.0, iters=1000)
                fd_score = _score(fd_pts)
                if fd_score > best_score:
                    best_pts = fd_pts
                    best_score = fd_score
            except Exception:
                pass

    # 4. Progressive soft-min polish from best
    polish_deadline = _DEADLINE - 35.0
    if time.time() < polish_deadline:
        try:
            polished_pts = _optimize_from_start(best_pts, polish_deadline)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass

    # 5. Local search to escape local optima
    ls_deadline = _DEADLINE - 20.0
    if time.time() < ls_deadline:
        try:
            ls_pts = _local_search(best_pts, ls_deadline)
            ls_score = _score(ls_pts)
            if ls_score > best_score:
                best_pts = ls_pts
                best_score = ls_score
        except Exception:
            pass

    # 6. Final high-precision polish with multiple restarts from perturbed points
    final_deadline = _DEADLINE - 5.0
    if time.time() < final_deadline:
        try:
            fine_betas = [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
            rng = np.random.RandomState(999)
            for attempt in range(4):
                if time.time() > final_deadline - 5.0:
                    break
                jitter = rng.uniform(-0.005, 0.005, best_pts.shape)
                perturbed = best_pts + jitter
                radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
                mask = radii > 1.0
                if np.any(mask):
                    perturbed[mask] = perturbed[mask] / radii[mask, np.newaxis]

                final_pts = _optimize_from_start(perturbed, final_deadline - 5.0, betas=fine_betas)
                final_score = _score(final_pts)
                if final_score > best_score:
                    best_pts = final_pts
                    best_score = final_score
        except Exception:
            pass

    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return initializations[best_init_idx]

    # Ensure all points are strictly inside
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] * 0.999 / radii[mask, np.newaxis]

    return best_pts
