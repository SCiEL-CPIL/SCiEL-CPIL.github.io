"""Packing 200 equal circles in a unit square using smooth surrogate optimization."""
import numpy as np
import time

N = 200
HALF = 0.5


def _soft_min(d, alpha):
    """Soft minimum of array d via log-sum-exp (stable)."""
    d = np.asarray(d, dtype=np.float64)
    m = np.min(d)
    if m > 1e12:
        return m
    # Use the negative log-sum-exp trick: softmin(d) = -logsumexp(-d/alpha)*alpha
    neg = -d / alpha
    mx = np.max(neg)
    s = np.sum(np.exp(neg - mx))
    if s <= 0.0:
        return m
    return -(mx + np.log(s)) * alpha


def _score(pts, alpha):
    """Smooth surrogate for the true packing radius.
    True r = min( min_{i<j} |d_ij|/2, min_i min_k (HALF - |c_ik|) )
    We compute soft-min over all pair half-distances and boundary distances.
    """
    pts = np.asarray(pts, dtype=np.float64)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    half_d = dist / 2.0
    # Boundary distances: HALF - |coord|
    bx = HALF - np.abs(pts[:, 0])
    by = HALF - np.abs(pts[:, 1])
    boundary = np.concatenate([bx, by])
    all_vals = np.concatenate([half_d.ravel(), boundary])
    return _soft_min(all_vals, alpha)


def _true_radius(pts):
    """Exact radius for a point set (for final validation/reporting)."""
    pts = np.asarray(pts, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    r_bound = np.min(HALF - np.abs(pts))
    return float(min(r_pair, r_bound))


def _gradient(pts, alpha):
    """Gradient of the soft-min surrogate w.r.t. point positions.
    softmin(d) = -alpha * log(sum(exp(-d/alpha)))
    d/d(pts) softmin = alpha * sum_j (exp(-d_j/alpha) / sum_k exp(-d_k/alpha)) * d(pts, d_j)
    where d(pts, d_j) is the gradient of distance d_j w.r.t. pts.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-16)
    np.fill_diagonal(dist, np.inf)
    half_d = dist / 2.0

    # Boundary terms
    # For each point i, boundary distance b_ik = HALF - |c_ik|
    # Gradient w.r.t. c_ik is -sign(c_ik)
    bx = HALF - np.abs(pts[:, 0])
    by = HALF - np.abs(pts[:, 1])
    boundary = np.concatenate([bx, by])

    all_vals = np.concatenate([half_d.ravel(), boundary])

    # Weights
    neg = -all_vals / alpha
    mx = np.max(neg)
    exps = np.exp(neg - mx)
    s = np.sum(exps)
    if s <= 0:
        return np.zeros_like(pts)
    w = exps / s  # weights sum to 1

    grad = np.zeros_like(pts)

    # Pair contributions: w over the N*(N-1) pair half-distances
    w_pairs = w[:n * (n - 1)].reshape(n, n - 1)
    # Reconstruct full pair weight matrix (symmetric)
    w_full = np.zeros((n, n))
    # half_d.ravel() is row-major: for i, then j != i
    # Actually half_d.ravel() includes both (i,j) and (j,i). Let's just use the full matrix.
    # Recompute weights on full matrix for clarity.
    neg_full = -half_d / alpha
    mx_full = np.max(neg_full)
    exps_full = np.exp(neg_full - mx_full)
    # Include boundary in normalization
    neg_b = -boundary / alpha
    exps_b = np.exp(neg_b - mx_full)
    s_full = np.sum(exps_full) + np.sum(exps_b)
    if s_full <= 0:
        return grad
    wf = exps_full / s_full
    wb = exps_b / s_full

    # Pair gradient: for each pair (i,j), d(half_d_ij)/d(pts_i) = (pts_i - pts_j)/dist_ij
    # since half_d = dist/2, d(half_d)/d(pts_i) = (pts_i - pts_j)/dist_ij
    # softmin gradient = alpha * sum w_j * d(d_j)/d(pts)
    # For point i: sum_j wf[i,j] * (pts_i - pts_j)/dist_ij
    inv_dist = np.zeros_like(dist)
    mask = dist < 1e12
    inv_dist[mask] = 1.0 / dist[mask]
    # diff[i,j] = pts_i - pts_j
    # contribution to grad[i] from pair (i,j): wf[i,j] * diff[i,j] * inv_dist[i,j]
    grad_pair = np.sum(wf[:, :, None] * diff * inv_dist[:, :, None], axis=1)
    grad += alpha * grad_pair

    # Boundary gradient: for point i, x-component: wb[2i] * (-sign(pts[i,0]))
    # b_x_i = HALF - |x_i|, db/dx_i = -sign(x_i)
    # softmin grad += alpha * wb[2i] * (-sign(x_i))
    sign_x = np.sign(pts[:, 0])
    sign_y = np.sign(pts[:, 1])
    grad[:, 0] += alpha * wb[0::2] * (-sign_x)
    grad[:, 1] += alpha * wb[1::2] * (-sign_y)

    return grad


def _make_initial(seed):
    """Generate an initial configuration with some structure."""
    rng = np.random.default_rng(seed)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    # Add small random perturbation to break symmetry
    pts += rng.normal(0, 0.002, size=pts.shape)
    # Clip to be safely inside
    pts[:, 0] = np.clip(pts[:, 0], -HALF + 0.001, HALF - 0.001)
    pts[:, 1] = np.clip(pts[:, 1], -HALF + 0.001, HALF - 0.01)
    pts[:, 1] = np.clip(pts[:, 1], -HALF + 0.001, HALF - 0.001)
    return pts


def _optimize(pts, deadline, rng):
    """Local optimization using gradient ascent on soft-min surrogate.
    Returns best points found."""
    best_pts = pts.copy()
    best_score = _score(pts, 0.01)

    # Annealing schedule for alpha
    alpha_start = 0.02
    alpha_end = 0.002
    lr = 0.0005

    t0 = time.time()
    step = 0
    while time.time() < deadline:
        # Animate alpha
        frac = min(1.0, step / 500.0)
        alpha = alpha_start * (alpha_end / alpha_start) ** frac

        g = _gradient(pts, alpha)
        # Gradient ascent (we want to maximize soft-min)
        pts_new = pts + lr * g
        # Keep inside square with margin
        pts_new[:, 0] = np.clip(pts_new[:, 0], -HALF + 1e-6, HALF - 1e-6)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -HALF + 1e-6, HALF - 1e-6)

        s_new = _score(pts_new, alpha)
        if s_new > best_score - 1e-9:
            pts = pts_new
            if s_new > best_score:
                best_score = s_new
                best_pts = pts_new.copy()
        else:
            # Line search / reject
            pts = pts  # keep old
            # occasional random kick to escape
            if rng.random() < 0.01:
                kick = rng.normal(0, 0.005, size=pts.shape)
                pts = pts + kick
                pts[:, 0] = np.clip(pts[:, 0], -HALF + 1e-6, HALF - 1e-6)
                pts[:, 1] = np.clip(pts[:, 1], -HALF + 1e-6, HALF - 1e-6)

        step += 1
        if step % 50 == 0:
            # Check time
            if time.time() > deadline:
                break

    return best_pts, best_score


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    rng = np.random.default_rng(12345)

    best_pts = None
    best_true_r = 0.0

    # Multiple restarts with different seeds
    seeds = list(range(0, 50))
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
        pts = _make_initial(seed)
        # Local optimize
        sub_deadline = min(deadline - 5.0, time.time() + 20.0)
        pts_opt, s_opt = _optimize(pts, sub_deadline, rng)
        r = _true_radius(pts_opt)
        if r > best_true_r:
            best_true_r = r
            best_pts = pts_opt.copy()

    if best_pts is None:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]

    # Final polish: a few more gradient steps with small alpha
    if time.time() < deadline - 2.0:
        alpha = 0.001
        lr = 0.0002
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            g = _gradient(best_pts, alpha)
            best_pts = best_pts + lr * g
            best_pts[:, 0] = np.clip(best_pts[:, 0], -HALF + 1e-6, HALF - 1e-6)
            best_pts[:, 1] = np.clip(best_pts[:, 1], -HALF + 1e-6, HALF - 1e-6)

    # Ensure validity: finite, inside
    best_pts = np.asarray(best_pts, dtype=np.float64)
    if not np.all(np.isfinite(best_pts)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pts
