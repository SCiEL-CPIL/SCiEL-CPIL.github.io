"""Portfolio search for Dzugutov PT-cluster global minima.

Strategy: start from geometrically informed polytetrahedral templates
(icosahedral shells at the equilibrium bond length) plus diverse random
seeds, then polish each with the provided gradient-based relaxer. A global
deadline guard returns the best valid structure with a safety margin.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# ---- global deadline guard (hard kill at 330 s; keep a margin) ----
_T0 = time.time()
_HARD = 330.0
_MARGIN = 25.0
_DEADLINE = _T0 + _HARD - _MARGIN


def _time_left():
    return _DEADLINE - time.time()


def _energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Exact Dzugutov pair energy (evaluator formula)."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    dpos = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(dpos, axis=2)
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    e = np.zeros_like(r)
    mask_a = r < a
    mask_b = (r < b) & (~mask_a)
    e[mask_a] = A * (r[mask_a] ** (-m) - B) * np.exp(c / (r[mask_a] - a))
    e[mask_b] = B * np.exp(d / (r[mask_b] - b))
    return float(e.sum())


def _valid(positions, min_dist=0.5):
    """Finite coords and pairwise separation >= min_dist."""
    p = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(p)):
        return False
    n = len(p)
    if n < 2:
        return True
    dpos = p[:, None, :] - p[None, :, :]
    d = np.linalg.norm(dpos, axis=2)
    iu = np.triu_indices(n, k=1)
    return bool(d[iu].min() >= min_dist)


# ---- icosahedral geometry (polytetrahedral building blocks) ----
def _icosa_unit():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    v /= np.linalg.norm(v[0])  # unit circumradius
    return v


_ICOSA = _icosa_unit()


def _icosa_shell(R):
    return R * _ICOSA


def _random_sphere(rng, R, n, min_dist=0.2):
    pts = []
    tries = 0
    while len(pts) < n and tries < 6000:
        tries += 1
        u = rng.normal(size=3)
        u /= (np.linalg.norm(u) + 1e-12)
        p = u * R
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


def _make_templates(n, rng):
    """Diverse PT-informed and random starting structures for n atoms."""
    L = 1.37  # equilibrium bond length
    R1 = L  # inner icosahedron (5 atoms)
    R2 = 2.0 * L  # 12-atom shell
    R3 = 2.6 * L  # 20-atom shell
    templates = []

    # (a) full icosahedron (13 atoms) + concentric shells, trimmed to n
    full = np.vstack([np.zeros((1, 3)), _icosa_shell(R1), _icosa_shell(R2)])
    if n <= len(full):
        templates.append(full[:n].copy())
    else:
        extra = _random_sphere(rng, R3, n - len(full), min_dist=0.3)
        templates.append(np.vstack([full, extra]))

    # (b) core + icosahedral shell (13 atoms) + random outer
    core = np.vstack([np.zeros((1, 3)), _icosa_shell(R1)])
    if n <= 13:
        templates.append(core[:n].copy())
    else:
        extra = _random_sphere(rng, R3, n - 13, min_dist=0.3)
        templates.append(np.vstack([core, extra]))

    # (c) icosahedral shell only (12 atoms) + random
    shell = _icosa_shell(R1)
    if n <= 12:
        templates.append(shell[:n].copy())
    else:
        extra = _random_sphere(rng, R3, n - 12, min_dist=0.3)
        templates.append(np.vstack([shell, extra]))

    # (d) compact random cluster
    radius = L * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    while len(pts) < n and tries < 20000:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * L for q in pts):
            pts.append(p)
    templates.append(np.array(pts))

    # (e) second random with different seed offset
    rng2 = np.random.default_rng(n + 12345)
    pts = []
    tries = 0
    while len(pts) < n and tries < 20000:
        tries += 1
        p = rng2.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * L for q in pts):
            pts.append(p)
    templates.append(np.array(pts))

    return templates


def _polish(positions, potential, parameter):
    """Best of a few gradient relaxations with light jitter."""
    base = np.asarray(positions, dtype=float)
    best_e = _energy(base)
    best_p = base.copy()
    rng = np.random.default_rng(abs(hash(tuple(np.round(base, 3)))) % (2**31))
    for _ in range(2):
        try:
            e, p = relax(base, potential, parameter)
            if _valid(p) and e < best_e:
                best_e, best_p = float(e), np.asarray(p, dtype=float)
        except Exception:
            pass
        # light jitter restart
        try:
            jit = base + rng.normal(0.0, 0.05, base.shape)
            e, p = relax(jit, potential, parameter)
            if _valid(p) and e < best_e:
                best_e, best_p = float(e), np.asarray(p, dtype=float)
        except Exception:
            pass
    return best_e, best_p


def construct(instance):
    size = int(instance["n"])
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size)

    templates = _make_templates(size, rng)
    best_e = np.inf
    best_p = templates[0]

    for idx, tmpl in enumerate(templates):
        if _time_left() < 8.0:  # need room for one more polish + return
            break
        if len(tmpl) != size:
            continue
        e, p = _polish(tmpl, potential, parameter)
        if _valid(p) and e < best_e:
            best_e, best_p = e, p

    if not _valid(best_p):
        # last-resort fallback: well-separated random points
        best_p = rng.uniform(-2.0, 2.0, (size, 3))

    return best_p
# EVOLVE-BLOCK-END
