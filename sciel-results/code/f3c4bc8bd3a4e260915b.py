"""Dzugutov polytetrahedral cluster global minima (fresh design)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# ---- global deadline guard (evaluator kills at 330 s) ----
_T0 = time.time()
_DEADLINE = 295.0          # seconds of budget we actually spend
_SAFETY = 15.0             # margin before the 330 s kill
_DEADLINE = _T0 + _DEADLINE

# ---- Dzugutov potential parameters ----
_A = 5.82
_B = 1.28
_a = 1.87
_b = 1.94
_c = 1.10
_d = 0.27
_m = 16.0


def _energy(pos):
    """Dzugutov potential energy (vectorised), plus a per-atom penalty if any
    pair is closer than 0.5.  Cheap (no gradient) for ranking candidates."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    d = d[iu]
    e = np.zeros_like(d)
    r1 = d < _a
    r2 = d < _b
    e[r1] = _A * (d[r1] ** (-_m) - _B) * np.exp(_c / (d[r1] - _a))
    r2 = r2 & ~r1
    e[r2] = _B * np.exp(_d / (d[r2] - _b))
    e = e.sum()
    if d.min() < 0.5:
        e += 50.0 * n
    return float(e)


def _grad(pos):
    return dzugutov_energy_gradient(pos)


def _polish(pos):
    """Local refinement: relax() then a tight L-BFGS-B with the analytic gradient."""
    try:
        pos = np.asarray(relax(pos, 'dzugutov'), dtype=float)
    except Exception:
        pass
    try:
        res = minimize(_energy, pos, jac=_grad, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.x is not None and np.all(np.isfinite(res.x)):
            pos = res.x
    except Exception:
        pass
    return pos


def _icosahedron():
    """Unit icosahedron vertices (12)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    v /= np.linalg.norm(v[0])
    return v


def _random_rotation(rng):
    # QR of a Gaussian matrix gives a uniform rotation
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.diag(R))
    return Q


def _polytetra(n, rng, r1, r2, r3, rot):
    """Nested Mackay icosahedra (13/55/147) trimmed to n atoms."""
    I = _icosahedron()
    I = I @ rot.T
    core = np.zeros((1, 3))
    shell1 = r1 * I
    shell2 = r2 * I
    shell3 = r3 * I
    # order: core, shell1, shell2, shell3  (1 + 12 + 42 + 120)
    cand = np.vstack([core, shell1, shell2, shell3])
    if n >= cand.shape[0]:
        return cand
    # keep the closest n to the centroid (core + inner shells first)
    c = cand.mean(axis=0)
    order = np.argsort(np.linalg.norm(cand - c, axis=1))[:n]
    return cand[order]


def construct(instance):
    global _DEADLINE
    n = int(instance["n"])
    rng = np.random.default_rng(1000 + n)
    best = None
    best_e = np.inf

    # number of diverse starts scales with cluster size / available time
    n_starts = {38: 16, 45: 16, 50: 20, 55: 20, 57: 24, 60: 24}.get(n, 20)

    for s in range(n_starts):
        if time.time() >= _DEADLINE:
            break
        rot = _random_rotation(rng)
        r1 = rng.uniform(1.02, 1.14)
        r2 = rng.uniform(1.78, 1.98)
        r3 = rng.uniform(2.62, 2.86)
        try:
            start = _polytetra(n, rng, r1, r2, r3, rot)
        except Exception:
            continue
        if start.shape[0] != n:
            continue
        try:
            pos = _polish(start)
        except Exception:
            continue
        if not np.all(np.isfinite(pos)):
            continue
        e = _energy(pos)
        if e < best_e:
            best_e = e
            best = pos
        # keep the overall best-so-far as the current incumbent for later starts
        if best is not None and s < n_starts - 1 and time.time() < _DEADLINE - 30.0:
            # occasional re-polish of the incumbent from a small perturbation
            pass

    # final safety polish of the best incumbent
    if best is not None and time.time() < _DEADLINE - 10.0:
        try:
            pos = _polish(best)
            e = _energy(pos)
            if e < best_e:
                best_e, best = e, pos
        except Exception:
            pass

    if best is None:
        # fallback: simple random compact cluster
        radius = 1.1 * n ** (1.0 / 3.0) * 0.9
        pts = []
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.9 for q in pts):
                pts.append(p)
        best = np.array(pts)
    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
