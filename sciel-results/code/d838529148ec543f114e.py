"""LJ38 cluster global minimum search.

Strategy (combination of seed structure + inspiration refinement):
  * Keep the seed's public interface: a single construct() -> (N,3) array.
  * Directly generate the known FCC truncated-octahedron geometry (the
    accepted global minimum for LJ38, E ~ -173.928427) by taking the 38
    nearest FCC lattice points to the origin and rescaling to the LJ
    equilibrium nearest-neighbour distance 2^(1/6).
  * Polish with L-BFGS using an analytic gradient (vectorised), plus a few
    alternative starts (icosahedral funnel, random-perturbed FCC) as a safety
    net in case the FCC seed is imperfect.
  * Explicit deadline guard: return the best configuration found before the
    120 s evaluation window (killed at 150 s), with a generous safety margin.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38
TARGET_NN = 2.0 ** (1.0 / 6.0)   # LJ pair-potential minimum (sigma = 1)
DEADLINE = 110.0                 # seconds; comfortably under the 120 s limit


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for an (N,3) array of positions."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]   # (N,N,3)
    r2 = np.einsum('ijk,ijk->ij', diff, diff)              # (N,N)
    np.fill_diagonal(r2, 1.0)                              # avoid /0 on diagonal
    r6 = r2 ** 3
    r12 = r6 * r6

    pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair, 0.0)
    energy = 0.5 * np.sum(pair)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)                  # (N,N)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)         # (N,1)
    grad = row_sum * pos - coeff @ pos
    return energy, grad


def _enforce_min_spacing(pos, min_d=0.5, tol=1e-3, max_iter=200):
    """Nudge any pair closer than min_d apart until all are >= min_d."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        D = cdist(pos, pos)
        np.fill_diagonal(D, np.inf)
        bad = (D < min_d - tol) & (np.arange(n)[:, None] < np.arange(n)[None, :])
        if not bad.any():
            break
        for i, j in zip(*np.where(bad)):
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.array([1.0, 0.0, 0.0])
                dist = 1.0
            push = (min_d + tol - dist) / 2.0
            pos[i] += d / dist * push
            pos[j] -= d / dist * push
    return pos


def _scale_to_target_nn(pts):
    """Scale so the average nearest-neighbour distance is TARGET_NN."""
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    nn = np.min(D, axis=1)
    avg = np.mean(nn)
    if avg > 1e-12:
        pts = pts * (TARGET_NN / avg)
    return pts


def _make_fcc_truncated_octahedron():
    """38 nearest FCC lattice points to the origin -> truncated octahedron."""
    max_coord = 3
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([float(i), float(j), float(k)])
    pts = np.array(pts, dtype=float)
    dists = np.sum(pts ** 2, axis=1)
    pts = pts[np.argsort(dists)][:N].copy()
    pts -= np.mean(pts, axis=0)
    return _scale_to_target_nn(pts)


def _make_icosahedral():
    """Icosahedral-funnel-like starting geometry (a local minimum trap)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    pts = [np.zeros(3)]
    for r in (1.0, 1.8, 2.6):
        pts.extend(ico * r)
    pts.append([0.3, 0.3, 0.3])
    pts = np.array(pts[:N], dtype=float)
    return _scale_to_target_nn(pts)


def _make_random_perturbed_fcc(seed):
    rng = np.random.default_rng(seed)
    base = _make_fcc_truncated_octahedron()
    return base + 0.1 * rng.standard_normal(base.shape)


def _relax(pos, deadline):
    """L-BFGS-B polish with analytic gradient; returns (positions, energy)."""
    x0 = pos.reshape(-1)

    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g

    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                       options={'maxiter': 2000, 'maxfun': 5000,
                                'maxls': 40, 'factr': 1e7, 'gtol': 1e-6})
        p = res.x.reshape(-1, 3)
        e = _lj_energy_and_grad(p)[0]
        return p, e
    except Exception:
        e = _lj_energy_and_grad(pos)[0]
        return pos, e


def construct():
    t0 = time.time()
    candidates = [
        _make_fcc_truncated_octahedron(),
        _make_icosahedral(),
        _make_random_perturbed_fcc(12345),
        _make_random_perturbed_fcc(98765),
    ]

    best_pos, best_e = None, np.inf
    for c in candidates:
        if time.time() - t0 > DEADLINE - 20.0:
            break
        pos, e = _relax(c, DEADLINE)
        if e < best_e:
            best_e, best_pos = e, pos
        if time.time() - t0 > DEADLINE - 10.0:
            break

    if best_pos is None:
        best_pos = candidates[0]

    best_pos = _enforce_min_spacing(best_pos)
    return np.asarray(best_pos, dtype=float)
