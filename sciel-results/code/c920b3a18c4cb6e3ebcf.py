"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    t0 = time.time()
    deadline = 135.0
    n = 8
    NP = 2 * n
    iu = np.triu_indices(n, 1)

    def pdist(pts):
        return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

    def F(x):
        pts = x.reshape(n, 2)
        d = pdist(pts)
        dd = np.maximum(d[iu], 1e-10)
        return float(np.sum(np.log(dd)) - 28.0 * np.log(np.max(dd)))

    def negF(x):
        return -F(x)

    # ---- candidate initializations ----
    k = np.arange(n)
    octagon = np.column_stack([np.cos(2 * np.pi * k / n),
                               np.sin(2 * np.pi * k / n)]).flatten()

    def hex_candidate(r):
        pts = np.zeros((n, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        ang = np.deg2rad([30, 90, 150, 210, 270, 330])
        for i, a in enumerate(ang):
            pts[2 + i] = [r * np.cos(a), r * np.sin(a)]
        return pts.flatten()

    candidates = [octagon, hex_candidate(0.85), hex_candidate(1.0),
                  hex_candidate(0.95)]

    # ---- global search with DE ----
    seed = 20240601
    for _ in range(120):
        if time.time() > deadline - 55:
            break
        try:
            de = differential_evolution(negF, [(-4, 4)] * NP, seed=seed,
                                        maxiter=60, popsize=14, tol=1e-9,
                                        polish=False, updating='deferred',
                                        mutation=(0.5, 1.5), recombination=0.9)
            if de.x is not None and np.all(np.isfinite(de.x)):
                candidates.append(de.x)
        except Exception:
            pass
        seed += 13

    # keep top distinct candidates by F
    candidates = sorted(candidates, key=lambda x: -F(x))
    top = []
    for c in candidates:
        if all(np.linalg.norm(c - t) > 0.4 for t in top):
            top.append(c)
        if len(top) >= 6:
            break

    # ---- scoring & polish ----
    def score_of(pts):
        d = pdist(pts)
        diam = np.max(d)
        if diam < 1e-12:
            return -1.0
        pts2 = pts / diam * 2.0
        d2 = pdist(pts2)
        P = float(np.prod(d2[iu]))
        return (P * P) / (n ** n)

    def neg_logprod(x):
        pts = x.reshape(n, 2)
        d = pdist(pts)
        dd = np.maximum(d[iu], 1e-10)
        return -float(np.sum(np.log(dd)))

    def cons(x):
        pts = x.reshape(n, 2)
        d = pdist(pts)
        return d[iu] - 2.0

    best_score = -1.0
    best_pts = octagon.reshape(n, 2)
    rng = np.random.default_rng(7)

    def polish(cand):
        nonlocal best_score, best_pts
        pts = cand.reshape(n, 2)
        d = pdist(pts)
        diam = np.max(d)
        if diam < 1e-12:
            return
        pts = pts / diam * 2.0
        inits = [pts]
        for _ in range(3):
            inits.append(pts + rng.normal(0, 0.04, size=(n, 2)))
        for init in inits:
            if time.time() > deadline - 8:
                return
            try:
                res = minimize(neg_logprod, init.flatten(), method='SLSQP',
                               constraints=[{'type': 'ineq', 'fun': lambda p: -cons(p)}],
                               options={'maxiter': 400, 'ftol': 1e-14})
                cand_pts = res.x.reshape(n, 2)
            except Exception:
                cand_pts = init
            for cp in (cand_pts, init):
                sc = score_of(cp)
                if sc > best_score:
                    best_score = sc
                    best_pts = cp.copy()

    for cand in top:
        if time.time() > deadline - 8:
            break
        polish(cand)

    best_pts = np.asarray(best_pts, dtype=float)
    d = pdist(best_pts)
    if np.min(d[iu]) < 1e-6:
        best_pts = octagon.reshape(n, 2)
    return best_pts.tolist()
# EVOLVE-BLOCK-END
