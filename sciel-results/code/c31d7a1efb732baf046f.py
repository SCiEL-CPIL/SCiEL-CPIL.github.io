"""Improved packing: smooth-surrogate gradient ascent + polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 120.0  # safety margin before 150s kill


def _softmin(vals, beta):
    """Soft-min approximation of min(vals) using log-sum-exp."""
    # soft-min = -soft-max(-vals)
    neg = -vals
    m = np.max(neg)
    if m > 1e12:
        return m
    exp_vals = np.exp(neg - m)
    log_sum = np.log(np.sum(exp_vals)) + m
    return -log_sum / beta


def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Container margin
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)


def _objective_and_grad(pts, beta):
    """
    Maximize soft-min of all constraints.
    Constraints: pair distances (i<j) and (1 - |c_i|) for each i.
    We minimize the negative soft-min.
    """
    n = len(pts)
    # Collect all constraint values
    # Pair distances: for i < j
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            vals.append(d)
    # Container margins
    for i in range(n):
        r = np.linalg.norm(pts[i])
        vals.append(1.0 - r)

    vals = np.array(vals)
    soft = _softmin(vals, beta)

    # Gradient: d(soft-min)/d(pts)
    # soft-min = -(1/beta) * log(sum(exp(-vals/beta)))
    # d(soft-min)/d(val_k) = (1/beta) * exp(-vals_k/beta) / sum(exp(-vals/beta))
    # = (1/beta) * exp(-(vals_k - min_val)/beta) / sum(exp(-(vals - min_val)/beta))
    # Actually let's just do it numerically-free with the formula:
    neg_vals = -vals
    m = np.max(neg_vals)
    exp_vals = np.exp(neg_vals - m)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # these sum to 1, weight_k = exp(-val_k/beta)/sum

    # Gradient of soft-min w.r.t. each val_k:
    # soft = -log(sum(exp(-vals/beta)))
    # d(soft)/d(val_k) = (1/beta) * exp(-val_k/beta) / sum(exp(-vals/beta))
    # But wait: soft = -(1/beta)*log(sum(exp(-beta*vals))) ... let me redo.
    # Let f = softmin(vals, beta) = -log(sum(exp(-vals/beta)))
    # df/dval_k = -(1/beta)*(-exp(-val_k/beta))/sum(exp(-vals/beta)) = (1/beta)*exp(-val_k/beta)/sum
    # Hmm, that's not right either. Let me be careful.
    # f = -log(S) where S = sum_k exp(-val_k/beta)
    # df/dval_k = -(1/S) * dS/dval_k = -(1/S) * (-1/beta)*exp(-val_k/beta) = (1/(beta*S))*exp(-val_k/beta)
    # So df/dval_k = weight_k / beta where weight_k = exp(-val_k/beta)/S

    # Now chain rule: val depends on pts.
    grad_pts = np.zeros_like(pts)

    # For pair distance d_ij = |p_i - p_j|:
    # d(d_ij)/d(p_i) = (p_i - p_j)/d_ij
    # d(d_ij)/d(p_j) = (p_j - p_i)/d_ij
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                diff = np.array([1e-8, 0.0])
                d = 1e-8
            # weight index: pairs come first, in order (i,j) for i<j
            # Index in vals: pair (i,j) is at index i*(2*n-i-1)//2 + j - i - 1 ... complex
            # Let me restructure: build index mapping
            pass

    # This is getting complex. Let me use a cleaner approach: build the full list and track indices.
    # Actually, let me just compute gradient numerically-free by restructuring.

    # Rebuild with index tracking
    val_list = []
    # (val, type, i, j) where type=0 means pair, type=1 means container
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            val_list.append((d, 0, i, j))
    for i in range(n):
        r = np.linalg.norm(pts[i])
        val_list.append((1.0 - r, 1, i, -1))

    vals = np.array([v[0] for v in val_list])
    neg_vals = -vals
    m = np.max(neg_vals)
    exp_vals = np.exp(neg_vals - m)
    S = np.sum(exp_vals)
    weights = exp_vals / S  # weight_k

    grad_pts = np.zeros_like(pts)
    for k, (val, typ, i, j) in enumerate(val_list):
        w = weights[k] / beta
        if typ == 0:  # pair distance
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                diff = np.array([1e-8, 0.0])
                d = 1e-8
            grad_pts[i] += w * diff / d
            grad_pts[j] -= w * diff / d
        else:  # container: val = 1 - |p_i|
            r = np.linalg.norm(pts[i])
            if r < 1e-12:
                grad_pts[i] += 0.0
            else:
                grad_pts[i] -= w * pts[i] / r  # d(1-|p|)/d(p) = -p/|p|

    # We want to maximize soft, so gradient ascent: pts += lr * grad
    # For scipy minimize, we minimize -soft, so return -soft and -grad
    loss = -soft
    grad = -grad_pts
    return loss, grad


def _gradient_ascent(pts, beta_start, beta_end, n_iters, lr, time_limit):
    """Run gradient ascent on the soft-surrogate objective."""
    pts = pts.copy()
    t0 = time.time()
    beta = beta_start
    for it in range(n_iters):
        if time.time() - t0 > time_limit:
            break
        # Anneal beta
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iters - 1, 1))
        loss, grad = _objective_and_grad(pts, beta)
        # Gradient step
        pts = pts + lr * grad
        # Project back into unit disk (hard constraint enforcement)
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask][:, None] * 0.999
    return pts


def _polish(pts, time_limit):
    """Nelder-Mead polish on the true nonsmooth objective."""
    t0 = time.time()
    current_score = _compute_score(pts)

    def neg_score(x):
        pts = x.reshape(-1, 2)
        s = _compute_score(pts)
        return -s

    x0 = pts.flatten()
    res = minimize(neg_score, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                            'adaptive': True})
    pts_opt = res.x.reshape(-1, 2)
    score_opt = _compute_score(pts_opt)
    if score_opt > current_score:
        return pts_opt
    return pts


def construct():
    t_start = time.time()
    time_budget = DEADLINE - (time.time() - t_start)

    # Multiple initializations
    best_pts = None
    best_score = -1.0

    # Init 1: Hexagonal lattice
    def hex_init(spacing):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Pick N points closest to center for better packing
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[np.argsort(radius, kind='stable')[:N]]
        return points

    # Init 2: Concentric rings
    def ring_init(n_rings=7):
        pts = [np.array([0.0, 0.0])]
        for ring in range(1, n_rings):
            r = ring * 0.13
            n_in_ring = min(6 * ring, N - len(pts))
            if n_in_ring <= 0:
                break
            angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
            for a in angles:
                pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
        pts = np.array(pts[:N])
        return pts

    # Init 3: Random
    def random_init(seed):
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-0.8, 0.8, size=(N, 2))
        # Push inside
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 0.9
        pts[mask] = pts[mask] / radii[mask][:, None] * 0.9
        return pts

    inits = [
        hex_init(0.25),
        hex_init(0.22),
        hex_init(0.28),
        ring_init(7),
        ring_init(8),
        random_init(42),
        random_init(123),
    ]

    for init_idx, init_pts in enumerate(inits):
        if time.time() - t_start > DEADLINE - 5:
            break
        if len(init_pts) < N:
            continue

        # Gradient ascent with annealing
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_opt = min(time_remaining * 0.6, 20.0)  # cap per init

        if time_for_opt < 2.0:
            break

        pts_opt = _gradient_ascent(init_pts, beta_start=5.0, beta_end=50.0,
                                   n_iters=300, lr=0.01, time_limit=time_for_opt)

        # Polish
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_polish = min(time_remaining * 0.3, 10.0)
        if time_for_polish > 1.0:
            pts_opt = _polish(pts_opt, time_for_polish)

        score = _compute_score(pts_opt)
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()

    if best_pts is None:
        # Fallback
        best_pts = hex_init(0.25)

    return best_pts
