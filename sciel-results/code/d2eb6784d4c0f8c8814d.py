"""Rewritten program: soft-penalty energy minimization with L-BFGS-B + multi-restart + polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hard_radius(pts):
    """Exact common radius, -inf if invalid."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(pts)):
        return -np.inf
    d0 = np.linalg.norm(pts, axis=1)
    rb = 1.0 - np.min(d0)
    if rb <= 0.0:
        return -np.inf
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    rp = 0.5 * np.sqrt(np.min(d2))
    return min(rb, rp)


def _soft_radius(pts, k=80.0):
    """Smooth surrogate of the hard radius (higher = better, differentiable).
    Uses a soft-min over boundary terms and pair terms."""
    pts = np.asarray(pts, dtype=float)
    d0 = np.linalg.norm(pts, axis=1)
    b_terms = 1.0 - d0  # boundary margin per point
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    p_terms = 0.5 * np.sqrt(d2)  # pair margin per pair
    # soft min: -ln(sum(exp(-k*x)))/k
    def softmin(x):
        m = np.min(x)
        return m - np.log(np.sum(np.exp(-k * (x - m)))) / k
    return min(softmin(b_terms), softmin(p_terms))


def _energy_and_grad(x, w):
    """Objective to MINIMIZE: overlap penalty + boundary penalty.
    x is flattened (N,2). Returns (f, g).
    Penalty: for each pair, if d < 2r_target then push; use r_target = current soft radius? 
    Simpler: minimize sum of squared negative margins.
    We minimize: sum_{i<j} max(0, 2r0 - d_ij)^2 + sum_i max(0, |c_i| - (1-r0))^2
    where r0 is a fixed target radius. To maximize radius, we increase r0 over time (anneal).
    """
    r0 = w[0]
    lam = w[1]
    pts = x.reshape(N, 2)
    # boundary
    d0 = np.linalg.norm(pts, axis=1)
    over_b = np.maximum(0.0, d0 - (1.0 - r0))
    f_b = np.sum(over_b ** 2)
    # gradient wrt pts: 2*over_b * (pts/d0)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit = pts / d0[:, None]
    unit = np.where(d0[:, None] < 1e-12, 0.0, unit)
    g_b = 2.0 * over_b[:, None] * unit
    # pairs
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    d = np.sqrt(np.maximum(d2, 1e-16))
    np.fill_diagonal(d, np.inf)
    # margin: we want d >= 2r0. penalty where d < 2r0
    neg = np.maximum(0.0, 2.0 * r0 - d)
    f_p = np.sum(neg ** 2)
    # gradient: d/dpts of sum_{i<j} neg_{ij}^2
    # neg_{ij} = max(0, 2r0 - d_{ij}); d neg/d d_{ij} = -1 if d<2r0 else 0
    # d d_{ij}/d pts_i = -(pts_j - pts_i)/d = (pts_i - pts_j)/d
    # So contribution to g_i += 2*neg * (-1) * d(neg)/d d * ... let's do carefully.
    # f_p = sum_{i<j} neg_{ij}^2
    # df_p/d pts_i = sum_{j!=i} 2*neg_{ij} * d neg/d pts_i
    # d neg_{ij}/d pts_i = -1 * d d_{ij}/d pts_i (when active) = -1 * (pts_i - pts_j)/d_{ij}
    # So df_p/d pts_i = sum_{j!=i} 2*neg_{ij} * (-(pts_i - pts_j)/d_{ij})  [active only]
    active = neg > 0
    # build gradient via accumulation
    g_p = np.zeros_like(pts)
    # vectorized: for each pair (i,j) active, add to g_i and g_j
    ii, jj = np.where(active)
    if len(ii) > 0:
        v = pts[ii] - pts[jj]  # pts_i - pts_j
        d_active = d[ii, jj]
        coeff = 2.0 * neg[ii, jj] / d_active  # this is 2*neg/d
        # g_i += -coeff * v ; g_j += +coeff * v
        np.add.at(g_p, ii, -coeff[:, None] * v)
        np.add.at(g_p, jj, coeff[:, None] * v)
    f = f_b + f_p
    g = (g_b + g_p).ravel()
    return f, g


def _make_hex_init(jitter=0.0, seed=0):
    """Hexagonal lattice inside unit disk, with optional jitter."""
    rng = np.random.default_rng(seed)
    # target: ~300 points, hex spacing s => area per point = s^2 * sqrt(3)/2
    # disk area pi; need ~300 points => s^2*sqrt3/2 ~ pi/300 => s ~ sqrt(2*pi/(300*sqrt3))
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    # build lattice
    pts_list = []
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - s * 0.6:
                pts_list.append([x, y])
    pts = np.array(pts_list)
    if len(pts) < N:
        # shrink spacing
        s *= 0.95
        pts_list = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - s * 0.6:
                    pts_list.append([x, y])
        pts = np.array(pts_list)
    # sort by radius, take N closest to center
    rad = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(rad, kind='stable')
    pts = pts[order[:N]]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
        # clamp
        d0 = np.linalg.norm(pts, axis=1)
        over = d0 > 0.98
        if np.any(over):
            pts[over] *= (0.98 / d0[over])[:, None]
    return pts


def _make_ring_init(seed=0):
    """Concentric rings initialization."""
    rng = np.random.default_rng(seed)
    # choose number of rings
    best_pts, best_r = None, -np.inf
    for n_rings in range(6, 13):
        # distribute points: ring i has ~ (i+1)*base
        base = max(1, int(N / (n_rings * (n_rings + 1) / 2)))
        counts = []
        remaining = N
        for i in range(n_rings):
            c = base * (i + 1)
            c = min(c, remaining)
            if c <= 0:
                c = remaining if remaining > 0 else 0
            counts.append(c)
            remaining -= c
        # adjust to sum to N
        diff = N - sum(counts)
        if diff != 0:
            counts[-1] += diff
        if any(c <= 0 for c in counts) or sum(counts) != N:
            continue
        pts = []
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            rr = (i + 1) / (n_rings + 0.3) * 0.95
            ang = np.linspace(0, 2 * np.pi, c, endpoint=False) + i * 0.3
            for a in ang:
                pts.append([rr * np.cos(a), rr * np.sin(a)])
        pts = np.array(pts[:N])
        if len(pts) < N:
            continue
        r = _hard_radius(pts)
        if r > best_r:
            best_r, best_pts = r, pts.copy()
    return best_pts if best_pts is not None else _make_hex_init(0.0, seed)


def _optimize_from(pts0, deadline, seed=0):
    """Anneal r0 upward, minimize energy with L-BFGS-B. Return best pts."""
    rng = np.random.default_rng(seed)
    best_pts = np.array(pts0, dtype=float, copy=True)
    best_r = _hard_radius(best_pts)
    if best_r <= 0:
        return None
    x = best_pts.ravel()
    # start r0 slightly below current hard radius
    r0 = best_r * 0.9
    lam = 1.0
    # annealing schedule: increase r0 toward target over time
    # target radius: best known ~0.05315; we try to reach near that
    target_r = 0.053
    n_iter_budget = 40  # max L-BFGS-B iterations per anneal step
    # number of anneal steps
    n_anneal = 25
    for step in range(n_anneal):
        if time.time() > deadline - 5.0:
            break
        # increase r0
        frac = (step + 1) / n_anneal
        r0 = best_r * (1 - frac) + target_r * frac
        r0 = min(r0, 0.0535)  # cap
        lam = 1.0 + 2.0 * frac
        w = np.array([r0, lam])
        res = minimize(_energy_and_grad, x, args=(w,), method='L-BFGS-B',
                       jac=True, options={'maxiter': n_iter_budget, 'gtol': 1e-8,
                                          'ftol': 1e-12, 'maxls': 20})
        if res.x is not None and np.all(np.isfinite(res.x)):
            pts = res.x.reshape(N, 2)
            # clamp inside
            d0 = np.linalg.norm(pts, axis=1)
            over = d0 > 0.999
            if np.any(over):
                pts[over] *= (0.999 / d0[over])[:, None]
            r = _hard_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
                x = pts.ravel()
        # small random kick to escape local minima
        if step % 5 == 4:
            x = x + rng.normal(0, 0.002 * best_r, size=x.shape)
            pts = x.reshape(N, 2)
            d0 = np.linalg.norm(pts, axis=1)
            over = d0 > 0.999
            if np.any(over):
                pts[over] *= (0.999 / d0[over])[:, None]
            x = pts.ravel()
    return best_pts


def _polish(pts, deadline):
    """Fine hill-climb on hard radius with small steps, moving bottleneck pair/point."""
    pts = np.array(pts, dtype=float, copy=True)
    cur = _hard_radius(pts)
    if cur <= 0:
        return None
    best_pts = pts.copy()
    best_r = cur
    step = 1e-4
    no_improve = 0
    max_no = 30
    while time.time() < deadline - 2.0 and no_improve < max_no:
        d0 = np.linalg.norm(pts, axis=1)
        rb = 1.0 - np.min(d0)
        ib = int(np.argmin(d0))
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        ii, jj = np.unravel_index(np.argmin(d2), d2.shape)
        d = np.sqrt(d2[ii, jj])
        rp = 0.5 * d
        if rp < rb:
            v = pts[jj] - pts[ii]
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                break
            v = v / nrm
            pts[ii] -= 0.5 * step * v
            pts[jj] += 0.5 * step * v
            revert = lambda: (setattr(pts, '__getitem__', None), None)  # placeholder
        else:
            p = pts[ib]
            nrm = np.linalg.norm(p)
            if nrm < 1e-12:
                break
            pts[ib] -= step * (p / nrm)
        # clamp
        d0 = np.linalg.norm(pts, axis=1)
        over = d0 > 0.9999
        if np.any(over):
            pts[over] *= (0.9999 / d0[over])[:, None]
        new_r = _hard_radius(pts)
        if new_r > cur:
            cur = new_r
            step = max(1e-6, step * 0.95)
            no_improve = 0
            if cur > best_r:
                best_r = cur
                best_pts = pts.copy()
        else:
            # revert
            if rp < rb:
                v = pts[jj] - pts[ii]
                nrm = np.linalg.norm(v)
                if nrm < 1e-12:
                    break
                v = v / nrm
                pts[ii] += 0.5 * step * v
                pts[jj] -= 0.5 * step * v
            else:
                p = pts[ib]
                nrm = np.linalg.norm(p)
                if nrm < 1e-12:
                    break
                pts[ib] += step * (p / nrm)
            step *= 0.7
            no_improve += 1
            if step < 1e-8:
                break
    return best_pts


def construct():
    start = time.time()
    deadline = start + 140.0  # 150s wall limit, 10s safety

    # Generate multiple initializations
    inits = []
    # hex lattice no jitter
    inits.append(_make_hex_init(0.0, seed=0))
    # hex with small jitter
    for s in range(1, 5):
        inits.append(_make_hex_init(0.003, seed=s))
    # ring init
    inits.append(_make_ring_init(seed=10))

    # Evaluate all, pick best few
    scored = []
    for i, pts in enumerate(inits):
        if pts is None or len(pts) != N:
            continue
        r = _hard_radius(pts)
        if r > 0:
            scored.append((r, i, pts))
    scored.sort(key=lambda x: -x[0])

    # Optimize top candidates within time budget
    best_pts = scored[0][2].copy()
    best_r = scored[0][0]

    # time per optimization: allocate
    n_opt = min(4, len(scored))
    per_time = (deadline - 20.0) / n_opt
    for idx in range(n_opt):
        r0, i0, pts0 = scored[idx]
        t_start = time.time()
        t_end = t_start + per_time
        res = _optimize_from(pts0, t_end, seed=100 + idx)
        if res is not None:
            r = _hard_radius(res)
            if r > best_r:
                best_r = r
                best_pts = res.copy()

    # Final polish on the best
    if time.time() < deadline - 3.0:
        polished = _polish(best_pts, deadline)
        if polished is not None:
            r = _hard_radius(polished)
            if r > best_r:
                best_r = r
                best_pts = polished

    # Validity guard
    if _hard_radius(best_pts) <= 0.0:
        return scored[0][2]
    return best_pts
# EVOLVE-BLOCK-END
