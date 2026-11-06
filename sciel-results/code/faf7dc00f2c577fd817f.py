"""Dzugutov cluster global minima: icosahedral-shell seeds + gradient (BFGS) relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 275.0  # safety margin under the 300 s kill

# Dzugutov parameters (used to build a fast vectorised energy for the line search)
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    ri = r[iu]
    e = np.zeros(ri.shape, dtype=float)
    m1 = ri < _a
    m2 = (ri >= _a) & (ri < _b)
    e[m1] = _A * (ri[m1] ** (-_m) - _B) * np.exp(_c / (ri[m1] - _a))
    e[m2] = _B * np.exp(_d / (ri[m2] - _b))
    return float(e.sum())


def _energy_fn(x):
    return _energy(x)


def _grad_fn(x):
    p = np.asarray(x, dtype=float).reshape(-1, 3)
    try:
        g = dzugutov_energy_gradient(p)
        g = np.asarray(g, dtype=float).reshape(-1)
        if np.all(np.isfinite(g)):
            return g
    except Exception:
        pass
    # finite-difference fallback
    n = p.size
    h = 1e-5
    e0 = _energy(x)
    g = np.empty(n, dtype=float)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (_energy(xp) - _energy(xm)) / (2 * h)
    return g


def _min_sep_ok(pos):
    if pos.shape[0] < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(pos.shape[0], k=1)
    return r[iu].min() >= _MIN_SEP - 1e-9


def _make_icosahedron(scale):
    """12 vertices of a regular icosahedron scaled to nearest-neighbour distance ~scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    # edge length of this icosahedron is 2
    v = v * (scale / 2.0)
    return v


def _icosahedral_seeds(n, base=1.10):
    """Generate several icosahedral-shell starting configurations for cluster size n."""
    seeds = []
    shells = [1, 12]  # core + icosahedron
    remaining = n - 13
    # Build candidate shell radii for the icosahedral family
    r1 = base
    r2 = base * 1.7320508  # ~sqrt(3)
    r3 = base * 2.0
    # place remaining atoms on successive shells using icosahedral directions
    dirs = _make_icosahedron(1.0)  # unit-ish directions
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    def build(radii):
        pts = [np.zeros(3)]
        # shell 1: 12
        for d in dirs:
            pts.append(d * r1)
        placed = 13
        idx = 0
        shell_id = 1
        while placed < n:
            shell_id += 1
            r = radii[min(shell_id, len(radii)) - 1]
            # distribute on a sphere: use icosahedral dirs plus radial jitter / higher shell
            for d in dirs:
                if placed >= n:
                    break
                # offset direction slightly per shell to avoid coincident sites
                off = 0.12 * (shell_id - 1)
                dd = d + off * np.array([0.0, 0.0, 1.0])
                dd = dd / (np.linalg.norm(dd) + 1e-12)
                pts.append(dd * r)
                placed += 1
            idx += 1
            if shell_id > 6:
                break
        return np.array(pts[:n])

    for r2f in (1.60, 1.70, 1.7320508, 1.85, 2.0):
        radii = [base, base * r2f, base * 2.0]
        try:
            cfg = build(radii)
            if cfg.shape[0] == n and _min_sep_ok(cfg):
                seeds.append(cfg)
        except Exception:
            pass
    # also a compact "random-shell" fallback
    rng = np.random.default_rng(n)
    for _ in range(2):
        pts = [np.zeros(3)]
        for i in range(n - 1):
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12)
            r = base * (1.0 + 0.5 * (i / max(1, n - 1)))
            pts.append(d * r)
        cfg = np.array(pts)
        if _min_sep_ok(cfg):
            seeds.append(cfg)
    return seeds


def _relax_bfgs(pos):
    """Gradient-based BFGS relaxation with line-search energy; returns (energy, positions)."""
    x0 = pos.reshape(-1)
    try:
        res = minimize(
            _energy_fn, x0, jac=_grad_fn,
            method="BFGS",
            options={"maxiter": 300, "gtol": 1e-5, "disp": False},
        )
        e = _energy(res.x)
        p = res.x.reshape(-1, 3)
        if _min_sep_ok(p):
            return e, p
    except Exception:
        pass
    return _energy(x0), pos


def _perturb(pos, scale):
    rng = np.random.default_rng(abs(hash(tuple(np.round(pos, 3)))) % (2**31))
    return pos + rng.normal(0.0, scale, size=pos.shape)


def construct(instance):
    n = instance["n"]
    potential = instance.get("potential", "dzugutov")
    param = instance.get("rho", instance.get("kind"))

    seeds = _icosahedral_seeds(n)
    if not seeds:
        # last-resort compact random
        rng = np.random.default_rng(n)
        rad = 1.1 * n ** (1.0 / 3.0)
        pts = []
        while len(pts) < n:
            p = rng.uniform(-rad, rad, 3)
            if np.linalg.norm(p) <= rad and all(np.linalg.norm(p - q) >= 0.8 for q in pts):
                pts.append(p)
        seeds = [np.array(pts)]

    best_e = np.inf
    best_pos = seeds[0]
    t0 = time.time()

    # primary: relax each seed with gradient BFGS
    for s in seeds:
        if _remaining() < 12.0:
            break
        try:
            e, p = _relax_bfgs(s)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass

    # secondary: perturbation restarts from the best structure
    for scale in (0.15, 0.30, 0.50):
        if _remaining() < 10.0:
            break
        try:
            e, p = _relax_bfgs(_perturb(best_pos, scale))
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass

    return np.asarray(best_pos, dtype=float)
# EVOLVE-BLOCK-END
