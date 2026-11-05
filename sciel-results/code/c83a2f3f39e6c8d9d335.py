"""Dzugutov cluster global minima: polytetrahedral seeds + aggressive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16.0

def _energy(pos):
    """Dzugutov potential energy (exact, for local checks only)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(n, 1)
    rs = r[iu]
    V = np.zeros_like(rs)
    m1 = rs < a
    m2 = (rs >= a) & (rs < b)
    if np.any(m1):
        V[m1] = A * (rs[m1] ** (-m) - B) * np.exp(c / (rs[m1] - a))
    if np.any(m2):
        V[m2] = B * np.exp(d / (rs[m2] - b))
    return float(np.sum(V))

def _fix_min_dist(pos, min_d=0.5):
    """Nudge overlapping atoms apart so all pairs are >= min_d apart."""
    n = len(pos)
    pos = pos.copy()
    for _ in range(2000):
        diff = pos[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        iu = np.triu_indices(n, 1)
        rs = r[iu]
        bad = rs < min_d
        if not np.any(bad):
            break
        bi = iu[0][bad]
        bj = iu[1][bad]
        for k in range(len(bi)):
            i, j = bi[k], bj[k]
            dv = pos[j] - pos[i]
            cur = np.linalg.norm(dv)
            if cur < 1e-9:
                dv = np.array([1.0, 0.0, 0.0])
                cur = 1.0
            need = min_d + 1e-4
            pos[j] += dv / cur * (need - cur)
            pos[i] -= dv / cur * (need - cur)
    return pos

def _relax_with_grad(pos, max_iter=3000, tol=1e-9):
    """Steepest-descent local minimiser using the provided gradient."""
    pos = np.asarray(pos, dtype=float).copy()
    best_e = _energy(pos)
    best_p = pos.copy()
    step = 0.01
    g = dzugutov_energy_gradient(pos)
    gn = np.linalg.norm(g)
    it = 0
    while it < max_iter and gn > tol:
        pos = pos - step * g
        pos = _fix_min_dist(pos)
        e = _energy(pos)
        if e < best_e:
            best_e, best_p = e, pos.copy()
        g = dzugutov_energy_gradient(pos)
        gn = np.linalg.norm(g)
        it += 1
    return best_e, best_p

def _icosahedron_vertices():
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, t, 1/t], [0, t, -1/t], [0, -t, 1/t], [0, -t, -1/t],
        [t, 1/t, 0], [t, -1/t, 0], [-t, 1/t, 0], [-t, -1/t, 0],
        [1/t, 0, t], [1/t, 0, -t], [-1/t, 0, t], [-1/t, 0, -t],
    ], dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

_ICO = _icosahedron_vertices()

def _icosahedral_shell(n):
    """Build n points as concentric icosahedral shells (polytetrahedral motif)."""
    pts = []
    remaining = n
    shell_r = 0.0
    while remaining > 0:
        take = min(12, remaining)
        shell_r += 1.05
        pts.extend(_ICO[:take] * shell_r)
        remaining -= take
    return np.array(pts)

def _random_cluster(n, seed):
    rng = np.random.default_rng(seed)
    radius = 1.1 * n ** (1.0 / 3.0) * 0.9
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.9 for q in pts):
            pts.append(p)
    return np.array(pts)

def construct(instance):
    n = instance["n"]
    start = time.time()
    deadline = start + 320.0
    rng = np.random.default_rng(n * 131 + 7)

    # Seed pool: polytetrahedral (icosahedral) structures + random compact.
    seeds = []
    for scale in (0.9, 1.0, 1.1):
        seeds.append(_icosahedral_shell(n) * scale)
    for s in range(6):
        seeds.append(_random_cluster(n, s + 1))

    best_e = np.inf
    best_p = None

    def try_struct(pos):
        nonlocal best_e, best_p
        pos = _fix_min_dist(pos)
        try:
            e, p = relax(np.array(pos), instance.get("potential", "dzugutov"),
                         instance.get("rho", instance.get("kind")))
        except Exception:
            e, p = _relax_with_grad(pos)
        if e < best_e:
            best_e, best_p = e, np.array(p)

    for pos in seeds:
        if time.time() > deadline:
            break
        try_struct(pos)

    # Iterative kick-and-relax from the best structure to escape local minima.
    kick = 0.25
    while time.time() < deadline - 10.0:
        pos = best_p + rng.normal(0.0, kick, best_p.shape)
        pos = _fix_min_dist(pos)
        try_struct(pos)
        # gradually grow kick to explore further, then refine
        kick = min(kick + 0.03, 0.9)

    # Fine-tuning kicks around the best structure.
    for k in (0.05, 0.02, 0.01):
        if time.time() > deadline - 5.0:
            break
        pos = _fix_min_dist(best_p + rng.normal(0.0, k, best_p.shape))
        try_struct(pos)

    # Final polish with the gradient-based local minimiser.
    if time.time() < deadline - 3.0:
        e, p = _relax_with_grad(best_p)
        if e < best_e:
            best_e, best_p = e, p

    best_p = _fix_min_dist(best_p)
    return best_p
# EVOLVE-BLOCK-END
