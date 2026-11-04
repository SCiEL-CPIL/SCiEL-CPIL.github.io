"""Pack 300 equal circles in the unit disk.

Strategy: smooth force-based maximization of the minimum clearance, with
multi-scale initialization, an annealing schedule, and a final high-precision
polish. A deadline guard returns the best valid configuration found.
"""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 300
DEADLINE = 150.0
MARGIN = 5.0
RNG_SEED = 12345


def _clearance(x):
    """Return (min_clearance, feasible_flag) for a configuration."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (N, 2) or not np.all(np.isfinite(x)):
        return 0.0, False
    d = np.linalg.norm(x, axis=1)
    bound = 1.0 - d
    if np.any(bound <= 0.0):
        return 0.0, False
    tree = cKDTree(x)
    dd, _ = tree.query(x, k=2)
    min_pair = dd[:, 1].min()
    if min_pair <= 0.0:
        return 0.0, False
    return min(0.5 * min_pair, bound.min()), True


def _objective(x, beta):
    """Smooth surrogate: soft-min of pairwise half-distances and boundary."""
    x = np.asarray(x, dtype=np.float64)
    tree = cKDTree(x)
    dd, _ = tree.query(x, k=2)
    pair = 0.5 * dd[:, 1]
    d = np.linalg.norm(x, axis=1)
    bound = 1.0 - d
    vals = np.concatenate([pair, bound])
    m = vals.min()
    s = m - np.logaddexp.reduce(m - vals)
    return s + m


def _forces(x, alpha, beta):
    """Repulsive force pushing each point toward larger local clearance."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    f = np.zeros_like(x)
    tree = cKDTree(x)
    dd, idx = tree.query(x, k=2)
    for i in range(n):
        j = idx[i, 1]
        v = x[i] - x[j]
        d = dd[i, 1]
        if d < 1e-12:
            v = np.array([1e-6, 0.0])
            d = 1e-6
        # repulsion scaled by softness
        w = np.exp(-beta * d)
        f[i] += (alpha / d) * w * v
    # boundary: push inward if close to edge
    r = np.linalg.norm(x, axis=1)
    close = r > 0.999
    if np.any(close):
        f[close] += (x[close] / r[close, None]) * (-alpha * 0.5 * (r[close] - 0.999))
    return f


def _init_hex():
    """Hexagonal lattice points inside the disk."""
    pts = []
    s = 0.075
    import math
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            px = (i + 0.5 * (j % 2)) * s
            py = j * s * math.sqrt(3.0) / 2.0
            if px * px + py * py <= (1.0 - s / 2) ** 2:
                pts.append((px, py))
    pts = np.array(pts)
    if len(pts) > N:
        r = np.linalg.norm(pts, axis=1)
        order = np.argsort(r, kind="stable")
        pts = pts[order[:N]]
    return pts


def _init_random():
    """Random points in disk (rejection sampling)."""
    rng = np.random.default_rng(RNG_SEED)
    pts = []
    while len(pts) < N:
        a = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(rng.uniform(0, 1))
        pts.append((r * np.cos(a), r * np.sin(a)))
    return np.array(pts)


def _relax(x, alpha0, n_steps, beta_start, beta_end):
    """Simple gradient-based relaxation with annealing."""
    x = x.copy()
    for it in range(n_steps):
        frac = it / max(n_steps - 1, 1)
        alpha = alpha0 * (1.0 - frac) + 0.001
        beta = beta_start + (beta_end - beta_start) * frac
        f = _forces(x, alpha, beta)
        # step with adaptive size
        step = 0.01
        x = x + step * f
        # project back inside disk
        r = np.linalg.norm(x, axis=1)
        mask = r > 1.0
        if np.any(mask):
            x[mask] = x[mask] / r[mask, None] * 0.999
    return x


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN

    # Start from hexagonal lattice
    x = _init_hex()
    if x.shape[0] < N:
        x = _init_random()

    best = _clearance(x)[0]
    best_x = x.copy()

    # Multi-scale relaxation
    alphas = [0.1, 0.05, 0.02, 0.01, 0.005]
    betas = [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0)]
    steps = [200, 300, 400, 500, 600]

    for a, (b0, b1), ns in zip(alphas, betas, steps):
        if time.time() > deadline:
            break
        x = _relax(x, a, ns, b0, b1)
        c, ok = _clearance(x)
        if ok and c > best:
            best = c
            best_x = x.copy()

    # Final polish: small steps, high beta
    if time.time() < deadline:
        x = _relax(best_x, 0.002, 300, 80.0, 150.0)
        c, ok = _clearance(x)
        if ok and c > best:
            best = c
            best_x = x.copy()

    return best_x


if __name__ == "__main__":
    pts = construct()
    r, ok = _clearance(pts)
    print("radius:", r, "feasible:", ok)
