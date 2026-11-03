import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQ3 = math.sqrt(3.0)
MIN_DIST = 2.0          # touching centers of unit hexagons
DEADLINE = 140.0        # safety margin before 150s kill
SAFETY = 8.0


def _centers_to_points(centers):
    """All 66 vertices of N unit hexagons (side 1), each at 0 rotation."""
    ang = np.arange(6) * math.pi / 3.0
    base = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    return centers[:, None, :] + base[None, :, :]


def _max_center_radius(centers):
    return float(np.max(np.linalg.norm(centers, axis=1)))


def _max_vertex_radius(centers):
    pts = _centers_to_points(centers)
    return float(np.max(np.linalg.norm(pts, axis=1)))


def _pair_violation(centers, p=8.0):
    """Smooth penalty for any pair closer than MIN_DIST."""
    n = centers.shape[0]
    d2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    viol = np.clip(MIN_DIST - d, 0.0, None)
    return float(np.sum(viol ** p))


def _objective(centers, p=8.0, w=2000.0):
    """Smooth surrogate: max center radius (power sum) + pair separation penalty."""
    r = np.linalg.norm(centers, axis=1)
    maxr = float(np.sum(r ** p) ** (1.0 / p))
    return maxr + w * _pair_violation(centers, p)


def _anneal_step(centers, T, step):
    c = centers + step * np.random.randn_like(centers)
    return c


def _simulated_anneal(centers, T0, T1, iters, deadline, step0):
    cur = centers.copy()
    cur_obj = _objective(cur)
    best = cur.copy()
    best_obj = cur_obj
    for it in range(iters):
        if time.time() > deadline - SAFETY:
            break
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))
        step = step0 * (0.5 + 0.5 * (T / T0))
        cand = _anneal_step(cur, T, step)
        cand_obj = _objective(cand)
        if cand_obj < cur_obj or np.random.rand() < math.exp(-(cand_obj - cur_obj) / max(T, 1e-9)):
            cur = cand
            cur_obj = cand_obj
            if cur_obj < best_obj:
                best = cur.copy()
                best_obj = cur_obj
    return best, best_obj


def _refine(centers, deadline):
    """Nelder-Mead refinement of a feasible-ish configuration."""
    res = minimize(
        _objective,
        x0=centers.ravel(),
        method='Nelder-Mead',
        options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True,
                 'maxfev': 20000},
        callback=lambda x: None,
    )
    return res.x.reshape(N, 2)


def _initial_guesses():
    """Several diverse starting configurations."""
    guesses = []
    # ring of 6 + center + 4 more (ring at radius ~2.2)
    ring = []
    for k in range(6):
        a = k * math.pi / 3.0
        ring.append([2.2 * math.cos(a), 2.2 * math.sin(a)])
    guesses.append(np.array([ring] + [[0.0, 0.0]]))
    guesses[-1] = np.array(ring + [[0.0, 0.0]] + [[1.1, 0.0], [-1.1, 0.0], [0.0, 1.1], [0.0, -1.1]])
    # two offset hex rings
    g2 = []
    for k in range(6):
        a = k * math.pi / 3.0
        g2.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
    g2.append([0.0, 0.0])
    for k in range(4):
        a = k * math.pi / 2.0 + math.pi / 4.0
        g2.append([1.0 * math.cos(a), 1.0 * math.sin(a)])
    guesses.append(np.array(g2))
    # random
    for seed in range(3):
        rng = np.random.RandomState(seed)
        guesses.append(rng.randn(N, 2) * 2.0)
    return guesses


def construct():
    start = time.time()
    deadline = DEADLINE
    np.random.seed(42)

    best_centers = None
    best_score = float('inf')

    for guess in _initial_guesses():
        if time.time() > deadline - SAFETY:
            break
        # simulated anneal
        ann, ann_obj = _simulated_anneal(guess, T0=1.5, T1=1e-4, iters=4000,
                                         deadline=deadline, step0=0.8)
        # Nelder-Mead refine
        if time.time() > deadline - SAFETY:
            centers = ann
        else:
            try:
                centers = _refine(ann, deadline)
            except Exception:
                centers = ann
        # score by true max vertex radius
        score = _max_vertex_radius(centers)
        # only accept if pairs are well separated (feasibility)
        if _pair_violation(centers, p=4.0) < 1e-3 and score < best_score:
            best_score = score
            best_centers = centers.copy()

    if best_centers is None:
        # fallback: ring
        ring = []
        for k in range(6):
            a = k * math.pi / 3.0
            ring.append([2.2 * math.cos(a), 2.2 * math.sin(a)])
        best_centers = np.array(ring + [[0.0, 0.0]] + [[1.1, 0.0], [-1.1, 0.0], [0.0, 1.1], [0.0, -1.1]])
        best_score = _max_vertex_radius(best_centers)

    inner = [[float(best_centers[i][0]), float(best_centers[i][1]), 0.0] for i in range(N)]
    S = best_score * 2.0 / SQ3   # S = 2*R/√3  where R = max vertex radius
    return inner, (0.0, 0.0), float(S), 0.0
