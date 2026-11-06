"""Rewritten packer for 50 equal circles in the unit disk."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
T_LIMIT = 150.0
T_SAFE = 132.0          # explicit safety margin before the 150 s kill


def _score_raw(points):
    """True packing radius: min(pairwise/2, 1-|c_i|)."""
    pts = np.asarray(points, dtype=float)
    b = 1.0 - np.linalg.norm(pts, axis=1)
    n = pts.shape[0]
    dmin = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        dmin = min(dmin, d.min())
    return float(min(b.min(), 0.5 * dmin))


def _softmin_objective(x, beta):
    """Smooth (log-sum-exp) soft-min of all boundary & half-pair distances.

    Vectorized over all N-1 distances per point -> no Python inner loops.
    """
    pts = x.reshape(-1, 2)
    n = pts.shape[0]
    vals = []
    for i in range(n):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        vals.append(0.5 * d)
    vals.append(1.0 - np.linalg.norm(pts, axis=1))
    v = np.concatenate(vals)
    vmax = v.max()
    return -(vmax - np.log(np.sum(np.exp(-(v - vmax)))) / beta)


def _constrained_objective(x, r):
    """Penalty form: maximize r with a hard penalty for violations."""
    pts = x.reshape(-1, 2)
    n = pts.shape[0]
    viol = 0.0
    for i in range(n):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        viol += np.sum(np.maximum(0.0, 2.0 * r - d))
    viol += np.sum(np.maximum(0.0, r + np.linalg.norm(pts, axis=1) - 1.0))
    return -(r - 500.0 * viol)


def _repulse_relax(pts, time_budget):
    """Physics relaxation: repulsive pairwise + wall forces, adaptive step.

    Pushes apart the closest pairs and the most-outer points, directly
    increasing the minimum separation.  Anneals the step size.
    """
    pts = pts.astype(float).copy()
    n = pts.shape[0]
    t0 = time.time()
    lr = 0.004
    it = 0
    best = _score_raw(pts)
    best_pts = pts.copy()
    while time.time() - t0 < time_budget:
        F = np.zeros_like(pts)
        for i in range(n):
            d = pts[i + 1:] - pts[i]
            dist = np.linalg.norm(d, axis=1)
            dist = np.maximum(dist, 1e-9)
            f = d / (dist ** 3)[:, None]
            F[i] -= f.sum(axis=0)
            for j in range(1, n):
                F[i + j] += f[j - 1]
            r = np.linalg.norm(pts[i])
            if r > 1e-9:
                F[i] -= pts[i] / (r ** 2)
        step = lr * F
        step = np.clip(step, -0.05, 0.05)
        cand = pts + step
        s = _score_raw(cand)
        if s >= best - 1e-12:
            pts = cand
            if s > best:
                best = s
                best_pts = pts.copy()
                lr *= 1.02
        else:
            lr *= 0.8
        if it % 200 == 0 and it > 0:
            lr *= 0.98
        it += 1
    return best_pts, best


def _lbfgs_from(pts, beta, maxiter, time_budget):
    """L-BFGS-B on the smooth soft-min objective (vectorized objective)."""
    x0 = pts.flatten().copy()
    t0 = time.time()
    try:
        res = minimize(_softmin_objective, x0, args=(beta,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14,
                                'gtol': 1e-10, 'maxfun': 4000})
    except Exception:
        return pts, _score_raw(pts)
    out = res.x.reshape(-1, 2)
    s = _score_raw(out)
    cur = _score_raw(pts)
    if time.time() - t0 > time_budget:
        return pts, cur
    return (out, s) if s >= cur else (pts, cur)


def _seed_hex(r0):
    """Hexagonal lattice, pick N points nearest the centre."""
    best = None
    best_r = -1.0
    for r in (r0, r0 * 0.95, r0 * 1.05, r0 * 1.10, r0 * 0.90):
        k = int(2.0 / r) + 2
        rows = [((i + 0.5 * (j % 2)) * r, j * r * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        P = np.array(rows)
        rad = np.linalg.norm(P, axis=1)
        m = rad <= 1.0 - r / 2.0
        P = P[m]; rad = rad[m]
        if len(P) < N:
            continue
        sel = P[np.argsort(rad, kind='stable')[:N]]
        s = _score_raw(sel)
        if s > best_r:
            best_r = s; best = sel
    return best


def _seed_rings():
    """Concentric ring seed."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    idx = 1
    ring = 1
    while idx < N:
        cnt = min(N - idx, 6 * ring)
        ang = np.linspace(0, 2 * np.pi, cnt, endpoint=False) + np.pi / (6 * ring)
        rad = 0.185 * ring
        P[idx:idx + cnt] = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        idx += cnt
        ring += 1
    return P


def _seed_random(seed):
    rng = np.random.RandomState(seed)
    a = rng.uniform(0, 2 * np.pi, N)
    r = np.sqrt(rng.uniform(0, 0.85, N))
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def construct():
    t0 = time.time()
    deadline = t0 + T_SAFE

    seeds = []
    hx = _seed_hex(0.20)
    if hx is not None:
        seeds.append(hx)
    seeds.append(_seed_rings())
    seeds.append(_seed_random(42))
    seeds.append(_seed_random(123))

    best_pts = seeds[0]
    best_score = _score_raw(seeds[0])

    # Phase 1: quick repulse relaxation to build a good starting point.
    for s in seeds:
        if time.time() > deadline:
            break
        rem = deadline - time.time()
        if rem < 5:
            break
        pts, sc = _repulse_relax(s, min(rem * 0.3, 8.0))
        if sc > best_score:
            best_score = sc; best_pts = pts

    # Phase 2: L-BFGS-B on the best repulsed config (annealed beta).
    rem = deadline - time.time()
    if rem > 8:
        for beta in (120.0, 40.0):
            if time.time() > deadline:
                break
            rem = deadline - time.time()
            if rem < 8:
                break
            pts, sc = _lbfgs_from(best_pts, beta, maxiter=300, time_budget=rem * 0.7)
            if sc > best_score:
                best_score = sc; best_pts = pts
            # polish with a short repulse pass
            if time.time() < deadline - 6:
                p2, s2 = _repulse_relax(pts, min(deadline - time.time() - 6, 5.0))
                if s2 > best_score:
                    best_score = s2; best_pts = p2

    # Phase 3: constrained L-BFGS-B (r as a variable) from the best config.
    rem = deadline - time.time()
    if rem > 10:
        r0 = max(best_score * 0.999, 0.1)
        x0 = np.concatenate([best_pts.flatten(), [r0]])
        t1 = time.time()
        try:
            res = minimize(_constrained_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14,
                                    'gtol': 1e-10, 'maxfun': 2000})
            cands = res.x[:-N].reshape(-1, 2)
            sc = _score_raw(cands)
            if sc > best_score and time.time() - t1 < rem:
                best_score = sc; best_pts = cands
        except Exception:
            pass

    # Phase 4: final short polish with repulse.
    rem = deadline - time.time()
    if rem > 3:
        p2, s2 = _repulse_relax(best_pts, min(rem - 1, 8.0))
        if s2 > best_score:
            best_score = s2; best_pts = p2

    return best_pts
