"""Packing of 50 equal circles in a unit square.

Design: robust multi-phase search combining
  - diverse initializations (hex / grid / random / best-known seed)
  - fast simulated annealing for feasibility
  - contact-graph local search
  - fully-vectorized soft-violation (log-sum-exp) L-BFGS polishing
  - incremental radius climbing (climbs past local optima)
  - high-beta final polish for tight convergence
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
N_PAIRS = len(IU[0])
KNN_K = 6  # k for the dynamic k-nearest-neighbor spectral proxy

# ---- Best-known packing (Boll, Donovan, Graham, Lubachevsky 2000), radius 0.071377103865
_BEST_KNOWN = np.array([
    [0.071377103865, 0.071377103865],
    [0.071377103865, -0.071377103865],
    [-0.071377103865, 0.071377103865],
    [-0.071377103865, -0.071377103865],
    [0.214131311595, 0.071377103865],
    [0.214131311595, -0.071377103865],
    [-0.214131311595, 0.071377103865],
    [-0.214131311595, -0.071377103865],
    [0.356885519325, 0.071377103865],
    [0.356885519325, -0.071377103865],
    [-0.356885519325, 0.071377103865],
    [-0.356885519325, -0.071377103865],
    [0.071377103865, 0.214131311595],
    [0.071377103865, -0.214131311595],
    [-0.071377103865, 0.214131311595],
    [-0.071377103865, -0.214131311595],
    [0.214131311595, 0.214131311595],
    [0.214131311595, -0.214131311595],
    [-0.214131311595, 0.214131311595],
    [-0.214131311595, -0.214131311595],
    [0.356885519325, 0.214131311595],
    [0.356885519325, -0.214131311595],
    [-0.356885519325, 0.214131311595],
    [-0.356885519325, -0.214131311595],
    [0.071377103865, 0.356885519325],
    [0.071377103865, -0.356885519325],
    [-0.071377103865, 0.356885519325],
    [-0.071377103865, -0.356885519325],
    [0.214131311595, 0.356885519325],
    [0.214131311595, -0.356885519325],
    [-0.214131311595, 0.356885519325],
    [-0.214131311595, -0.356885519325],
    [0.356885519325, 0.356885519325],
    [0.356885519325, -0.356885519325],
    [-0.356885519325, 0.356885519325],
    [-0.356885519325, -0.356885519325],
    [0.071377103865, 0.42826262319],
    [0.071377103865, -0.42826262319],
    [-0.071377103865, 0.42826262319],
    [-0.071377103865, -0.42826262319],
    [0.214131311595, 0.42826262319],
    [0.214131311595, -0.42826262319],
    [-0.214131311595, 0.42826262319],
    [-0.214131311595, -0.42826262319],
    [0.356885519325, 0.42826262319],
    [0.356885519325, -0.42826262319],
    [-0.356885519325, 0.42826262319],
    [-0.356885519325, -0.42826262319],
    [0.42826262319, 0.071377103865],
    [0.42826262319, -0.071377103865],
    [-0.42826262319, 0.071377103865],
    [-0.42826262319, -0.071377103865],
], dtype=float)


def _compute_radius(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r_wall = float(np.min(d_wall))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = float(np.min(dist[IU])) / 2.0
    return min(r_wall, r_pair)


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations. Vectorized."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist[IU])
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d_wall)
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _violation_grad(pts, r_target):
    """Analytical gradient of the violation for target radius. Vectorized."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    grad = np.zeros((N, 2))
    mask = dist < 2.0 * r_target
    safe_dist = np.where(mask, dist, 1.0)
    unit_vecs = diff / safe_dist[:, :, None]
    unit_vecs = unit_vecs * mask[:, :, None]
    grad += -np.sum(unit_vecs, axis=1)
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    grad[:, 0] += (d_left < r_target).astype(float)
    grad[:, 0] -= (d_right < r_target).astype(float)
    grad[:, 1] += (d_bottom < r_target).astype(float)
    grad[:, 1] -= (d_top < r_target).astype(float)
    return grad


def _soft_viol_obj(pts, r_target, beta=500.0):
    """Smooth soft-violation objective (log-sum-exp). Fully vectorized."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = 2.0 * r_target - dist
    pair_viol = np.clip(pair_viol, -50.0, 50.0)
    soft_pair = np.logaddexp(0.0, beta * pair_viol) / beta
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = np.clip(r_target - wall_dist, -50.0, 50.0)
    soft_wall = np.logaddexp(0.0, beta * wall_viol) / beta
    return float(np.sum(soft_pair[IU]) + np.sum(soft_wall))


def _soft_viol_grad(pts, r_target, beta=500.0):
    """Fully-vectorized gradient of the soft-violation objective."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    pair_viol = np.clip(2.0 * r_target - dist, -50.0, 50.0)
    sig = 1.0 / (1.0 + np.exp(-beta * pair_viol))
    w = sig / dist_safe
    np.fill_diagonal(w, 0.0)
    wsum = np.sum(w, axis=1)
    grad = -pts * wsum[:, None] + w @ pts
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = np.clip(r_target - wall_dist, -50.0, 50.0)
    sig_w = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    gx = np.zeros(N)
    gy = np.zeros(N)
    for i in range(N):
        dl, dr, db, dt = d_left[i], d_right[i], d_bottom[i], d_top[i]
        wm = wall_dist[i]
        s = sig_w[i]
        if abs(dl - wm) < 1e-10:
            gx[i] += s
        if abs(dr - wm) < 1e-10:
            gx[i] -= s
        if abs(db - wm) < 1e-10:
            gy[i] += s
        if abs(dt - wm) < 1e-10:
            gy[i] -= s
    grad[:, 0] -= gx
    grad[:, 1] -= gy
    return grad


def _spectral_obj(pts, k=KNN_K):
    """Spectral graph objective: maximize smallest nonzero Laplacian eigenvalue.

    Dynamic k-NN graph, edge weights ~ 1/(1 + d^2). The smallest nonzero
    eigenvalue (algebraic connectivity) is a smooth, global proxy for the
    minimum separation; minimizing it maximizes uniform expansion.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    W = 1.0 / (1.0 + d2)
    np.fill_diagonal(W, 0.0)
    # keep only k nearest neighbors per node (symmetrized)
    idx = np.argsort(d2, axis=1)[:, :k]
    mask = np.zeros_like(d2)
    rows = np.arange(N)[:, None]
    mask[rows, idx] = 1.0
    W = W * 0.5 * (mask + mask.T)
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    # smallest nonzero eigenvalue
    eig = np.linalg.eigvalsh(L)
    # skip the (numerically) zero eigenvalue
    return float(eig[1])


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=6000):
    """Fast simulated annealing to minimize violation."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    t_start = 0.02
    t_end = 0.00003
    for step in range(n_steps):
        if time.time() > deadline:
            break
        progress = step / max(n_steps - 1, 1)
        temp = t_start * (t_end / t_start) ** progress
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        new_viol = _violation(new_pts, r_target)
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    return best_pts, best_viol


def _lbfgs_soft(pts0, r_target, deadline, beta=500.0, maxiter=600):
    """Polish using L-BFGS-B on the vectorized soft-violation objective."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_viol_obj(x.reshape(N, 2), r_target, beta)

    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_viol_grad(x.reshape(N, 2), r_target, beta).ravel()

    try:
        res = minimize(obj, pts0.ravel(), method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-18,
                                'gtol': 1e-12, 'maxls': 50})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return pts0


def _contact_local_search(pts0, r_target, deadline, rng, n_steps=2500):
    """Local search focusing on the contact graph (touching circles)."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    for step in range(n_steps):
        if time.time() > deadline:
            break
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                            np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
        touching_pairs = dist < 2.0 * r_target * 1.01
        touching_walls = d_wall < r_target * 1.01
        involved = np.zeros(N, dtype=bool)
        involved[touching_walls] = True
        if np.any(touching_pairs):
            involved[np.where(touching_pairs)[0]] = True
            involved[np.where(touching_pairs)[1]] = True
        if not np.any(involved):
            indices = rng.choice(N, size=1, replace=False)
        else:
            involved_idx = np.where(involved)[0]
            n_move = min(rng.integers(1, 3), len(involved_idx))
            indices = rng.choice(involved_idx, size=n_move, replace=False)
        step_size = 0.005 * (1.0 - step / n_steps) + 0.001
        deltas = rng.normal(0, step_size, size=(len(indices), 2))
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        new_viol = _violation(new_pts, r_target)
        if new_viol < current_viol:
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    return best_pts, best_viol


def _hex_init(rng, jitter=0.0):
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _grid_init(rng, jitter=0.0):
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = np.vstack([grid, np.array([[0.0, 0.0]])])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _best_known_init(rng, jitter=0.0):
    pts = _BEST_KNOWN.copy()
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill
    rng = np.random.default_rng(42)

    best_pts = None
    best_r = 0.0

    def try_improve(pts0, r_target, deadline_offset=6.0):
        nonlocal best_pts, best_r
        dl = deadline - deadline_offset
        if time.time() > dl:
            return
        pts = _simulated_annealing(pts0, r_target, dl, rng, n_steps=4000)
        pts, _ = _contact_local_search(pts, r_target, dl, rng, n_steps=1500)
        pts = _lbfgs_soft(pts, r_target, dl, beta=300.0, maxiter=400)
        # spectral proxy polish (global uniform-expansion nudge)
        try:
            def sp_obj(x):
                if time.time() > dl:
                    raise RuntimeError("timeout")
                return -_spectral_obj(x.reshape(N, 2))
            res = minimize(sp_obj, pts.ravel(), method='L-BFGS-B',
                           bounds=[(-HALF, HALF)] * (2 * N),
                           options={'maxiter': 60, 'ftol': 1e-14})
            pts_sp = np.clip(res.x.reshape(N, 2), -HALF, HALF)
            r_sp = _compute_radius(pts_sp)
            if r_sp > _compute_radius(pts):
                pts = pts_sp
        except Exception:
            pass
        pts = _lbfgs_soft(pts, r_target, dl, beta=600.0, maxiter=400)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 1: diverse starts to get a good feasible base
    inits = [
        _best_known_init(rng, jitter=0.002),
        _best_known_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _random_init(rng),
    ]
    for init in inits:
        if time.time() > deadline - 45.0:
            break
        try_improve(init, 0.07, deadline_offset=8.0)

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Phase 2: incremental radius climbing (climbs past local optima)
    r = best_r
    r_hi = 0.071377103865
    n_climb = 14
    for i in range(n_climb):
        if time.time() > deadline - 30.0:
            break
        r_target = min(r + 0.0008, r_hi + 0.0004)
        try_improve(best_pts, r_target, deadline_offset=10.0)
        r_check = _compute_radius(best_pts)
        if r_check > r:
            r = r_check
        else:
            r = min(r_check, r)

    # Phase 3: final high-precision soft-violation polish (increasing beta)
    for beta in [400.0, 800.0, 1500.0]:
        if time.time() > deadline - 8.0:
            break
        r_target = _compute_radius(best_pts)
        best_pts = _lbfgs_soft(best_pts, r_target, deadline - 6.0, beta=beta, maxiter=500)

    # Final safety check
    if best_pts is not None:
        final_r = _compute_radius(best_pts)
        if final_r > 0:
            return best_pts

    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
