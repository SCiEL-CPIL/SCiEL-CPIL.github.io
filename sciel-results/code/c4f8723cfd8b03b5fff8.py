# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 600.0  # safety margin before 630s kill


def _soft_min_obj(points, beta):
    """Return (value, per-point gradient) of soft-min of squared pairwise distances.
    value = log-sum-exp(-beta * d^2)  (larger = better, closer to log(91) at optimum)
    """
    n = points.shape[0]
    d2 = np.einsum('ij,ij->i', points, points)[:, None] + \
         np.einsum('ij,ij->i', points, points)[None, :] - \
         2.0 * (points @ points.T)
    # zero diagonal
    d2 = np.where(np.eye(n, dtype=bool), 1e12, d2)
    d2 = np.maximum(d2, 1e-12)
    # off-diagonal mask
    mask = ~np.eye(n, dtype=bool)
    x = -beta * d2
    x = np.where(mask, x, -1e12)
    xmax = x.max(axis=None)
    s = np.exp(x - xmax)
    s = np.where(mask, s, 0.0)
    logsumexp = xmax + np.log(s.sum())
    # gradient of logsumexp w.r.t. points
    # d(logsumexp)/d(points_k) = sum_j w_j * d(-beta*d2_kj)/d(points_k)
    # w_j = s_j / sum(s)  (normalized weights over all pairs)
    w = s / s.sum()  # shape (n,n), w_kj
    # d2_kj = |p_k - p_j|^2 ; d/dp_k = -2*(p_k - p_j)
    # grad_k = sum_j w_kj * (-beta) * (-2)*(p_k - p_j) = 2*beta*sum_j w_kj*(p_k - p_j)
    # = 2*beta*(p_k * sum_j w_kj - sum_j w_kj * p_j)
    row_sum = w.sum(axis=1)  # (n,)
    weighted = w @ points    # (n,n) @ (n,3) = (n,3), k-th row = sum_j w_kj p_j
    grad = 2.0 * beta * (points * row_sum[:, None] - weighted)
    return logsumexp, grad


def _optimize(points0, beta_start, beta_end, iters, rng, deadline):
    points = points0.copy().astype(float)
    # normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    points = points / norms

    best_val = -np.inf
    best_points = points.copy()

    for it in range(iters):
        if time.time() > deadline:
            break
        beta = beta_start + (beta_end - beta_start) * (it / max(1, iters - 1))
        val, grad = _soft_min_obj(points, beta)
        if val > best_val:
            best_val = val
            best_points = points.copy()

        # adaptive step: try a step, keep if improves
        # use a base step scaled by beta
        base_lr = 0.02 / (1.0 + 0.1 * beta)
        # per-point adaptive learning rates
        if not hasattr(_optimize, '_lr'):
            _optimize._lr = np.ones(points.shape[0]) * base_lr
        lr = _optimize._lr

        cand = points + lr[:, None] * grad
        # project back to sphere
        cn = np.linalg.norm(cand, axis=1, keepdims=True)
        cn[cn < 1e-12] = 1.0
        cand = cand / cn

        cval, _ = _soft_min_obj(cand, beta)
        if cval > val:
            _optimize._lr = np.minimum(_optimize._lr * 1.05, 0.2)
            points = cand
        else:
            _optimize._lr = _optimize._lr * 0.7
            # try a smaller step
            cand2 = points + (lr[:, None] * 0.3) * grad
            cn2 = np.linalg.norm(cand2, axis=1, keepdims=True)
            cn2[cn2 < 1e-12] = 1.0
            cand2 = cand2 / cn2
            cval2, _ = _soft_min_obj(cand2, beta)
            if cval2 > val:
                points = cand2

    return best_points, best_val


def _seed_icosahedron_poles(rng):
    """14 points: 12 icosahedron vertices + 2 poles (slightly perturbed)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=float)
    pts = np.vstack([verts, poles])
    # small random perturbation to break symmetry and avoid degenerate start
    pts = pts + 0.05 * rng.standard_normal(pts.shape)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _score(points):
    from scipy.spatial.distance import pdist
    d = pdist(points)
    if d.size == 0:
        return 0.0
    dmin, dmax = d.min(), d.max()
    if dmax <= 0:
        return 0.0
    return (dmin / dmax) ** 2


def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    deadline = _DEADLINE - 30.0  # 30s safety margin

    rng = np.random.default_rng(12345)

    best_points = None
    best_score = -1.0

    # Start 1: symmetric icosahedron + poles
    starts = [_seed_icosahedron_poles(rng)]
    # Additional random starts
    for _ in range(8):
        p = rng.standard_normal((n, 3))
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)

    for si, p0 in enumerate(starts):
        if time.time() > deadline:
            break
        # reset per-start adaptive lr
        _optimize._lr = np.ones(n) * 0.02
        # anneal beta
        iters = 400
        beta_start = 5.0
        beta_end = 60.0
        pts, val = _optimize(p0, beta_start, beta_end, iters, rng, deadline)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()

    if best_points is None:
        # fallback: random
        best_points = rng.standard_normal((n, 3))

    # Ensure finite and shape
    best_points = np.asarray(best_points, dtype=float)
    best_points[~np.isfinite(best_points)] = 0.0
    if best_points.shape != (14, 3):
        best_points = rng.standard_normal((14, 3))

    return best_points


# EVOLVE-BLOCK-END
