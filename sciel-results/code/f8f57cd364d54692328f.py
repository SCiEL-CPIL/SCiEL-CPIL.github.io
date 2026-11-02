"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Combined strategy:
  * Fast vectorized scoring over all 286 triples (from current program).
  * Symmetric ring parametrization (from inspiration): the heavy global search
    and most local refinement live in a low-dim ring space (radii+phase per ring),
    which is far easier to optimize and lets us afford many diverse starts.
  * Final full 26-D polish (L-BFGS-B + Nelder-Mead) on the best ring-built points
    to break symmetry and squeeze out the last gain.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 142.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Pre-compute all triples for fast vectorized scoring
# ---------------------------------------------------------------------------
_ti = []
_tj = []
_tk = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _ti.append(_i)
            _tj.append(_j)
            _tk.append(_k)
_ti = np.array(_ti, dtype=np.int32)
_tj = np.array(_tj, dtype=np.int32)
_tk = np.array(_tk, dtype=np.int32)


def _score_points(pts):
    """min triangle area / convex hull area (0.0 on any failure)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    cross = np.abs(
        (x[_tj] - x[_ti]) * (y[_tk] - y[_ti])
        - (y[_tj] - y[_ti]) * (x[_tk] - x[_ti])
    )
    min_area = 0.5 * np.min(cross)
    if min_area < 1e-15:
        return 0.0
    return float(min_area / hull_area)


def _neg_score(params):
    """Negative score for minimization (26-D free params)."""
    return -_score_points(params.reshape(N, 2))


def _build_from_rings(ring_params):
    """ring_params: list of (count, radius, phase). Returns (N,2) array."""
    parts = []
    total = 0
    for count, radius, phase in ring_params:
        if count == 0:
            continue
        if count == 1:
            ang = phase
            parts.append(np.array([[radius * np.cos(ang), radius * np.sin(ang)]]))
        else:
            angs = phase + 2 * np.pi * np.arange(count) / count
            parts.append(radius * np.column_stack([np.cos(angs), np.sin(angs)]))
        total += count
    if total != N:
        # pad/truncate to N for safety
        pts = np.vstack(parts) if parts else np.zeros((0, 2))
        if pts.shape[0] < N:
            pad = np.zeros((N - pts.shape[0], 2))
            pts = np.vstack([pts, pad])
        else:
            pts = pts[:N]
        return pts
    return np.vstack(parts)


def _ring_bounds(ring_counts, lo=0.0, hi=2.0):
    b = []
    for _ in ring_counts:
        b.append((lo, hi))   # radius (clamped non-negative at build)
        b.append((-np.pi, np.pi))  # phase
    return b


def _ring_objective(params, ring_counts):
    rp = []
    for i, c in enumerate(ring_counts):
        r = abs(params[2 * i])
        ph = params[2 * i + 1]
        rp.append((c, r, ph))
    return -_score_points(_build_from_rings(rp))


def _ring_init(ring_counts, trial, rng):
    k = len(ring_counts)
    if trial == 0:
        x = np.zeros(2 * k)
        for i in range(k):
            x[2 * i] = 1.0 / (i + 1)
        return x
    if trial == 1:
        x = np.zeros(2 * k)
        x[0] = 1.0
        return x
    x = rng.uniform(0.1, 1.6, size=2 * k)
    x[0::2] = np.abs(x[0::2])
    return x


def _optimize_ring(ring_counts, rng, max_iter=200, n_tries=8):
    """L-BFGS-B over ring params from several starts; returns (pts, score)."""
    bounds = _ring_bounds(ring_counts)
    best_pts, best_score = None, -1e18
    for trial in range(n_tries):
        x0 = _ring_init(ring_counts, trial, rng)
        try:
            res = minimize(
                _ring_objective, x0, args=(ring_counts,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': 6000, 'ftol': 1e-13},
            )
            pts = _build_from_rings(
                [(c, abs(res.x[2 * i]), res.x[2 * i + 1])
                 for i, c in enumerate(ring_counts)]
            )
            s = _score_points(pts)
            if s > best_score:
                best_score, best_pts = s, pts
        except Exception:
            pass
    return best_pts, best_score


def _de_ring(ring_counts, rng, maxiter=18, popsize=10):
    """Differential evolution over the ring space; returns (pts, score)."""
    bounds = _ring_bounds(ring_counts, lo=0.0, hi=2.0)
    try:
        res = differential_evolution(
            _ring_objective, bounds, args=(ring_counts,),
            seed=int(rng.randint(1, 10**6)),
            maxiter=maxiter, popsize=popsize, tol=1e-5,
            mutation=(0.5, 1.0), recombination=0.7,
            polish=False, init='sobol', updating='deferred', workers=1,
        )
        pts = _build_from_rings(
            [(c, abs(res.x[2 * i]), res.x[2 * i + 1])
             for i, c in enumerate(ring_counts)]
        )
        s = _score_points(pts)
        return pts, s
    except Exception:
        return None, -1e18


def _nm_ring(ring_counts, pts, rng, max_iter=400):
    """Nelder-Mead polish on ring params seeded by an existing point set.

    We recover approximate ring radii/phases from pts by clustering on radius,
    then polish. Returns (pts, score).
    """
    # Recover a ring param vector consistent with the structure from pts.
    # Simple approach: use least-squares-free heuristic - just seed from a
    # fresh random init and let NM explore; but better to seed from current best.
    # We instead build a param vector from current best by matching radii.
    k = len(ring_counts)
    # estimate radius per ring by sorting points by distance and splitting
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    x0 = np.zeros(2 * k)
    idx = 0
    for i, c in enumerate(ring_counts):
        grp = order[idx:idx + c]
        x0[2 * i] = float(np.mean(d[grp])) if c else 0.0
        # phase: mean angle of the group
        if c >= 2:
            angs = np.arctan2(pts[grp, 1], pts[grp, 0])
            x0[2 * i + 1] = float(np.mean(np.exp(1j * angs)).real and np.angle(np.sum(np.exp(1j * angs))))
        elif c == 1:
            x0[2 * i + 1] = float(np.arctan2(pts[grp[0], 1], pts[grp[0], 0]))
        idx += c
    try:
        res = minimize(
            _ring_objective, x0, args=(ring_counts,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-13,
                     'adaptive': True},
        )
        pts2 = _build_from_rings(
            [(c, abs(res.x[2 * i]), res.x[2 * i + 1])
             for i, c in enumerate(ring_counts)]
        )
        s = _score_points(pts2)
        return pts2, s
    except Exception:
        return pts, _score_points(pts)


def _full_polish(pts, rng, max_iter=300):
    """Full 26-D L-BFGS-B polish; returns improved pts."""
    try:
        res = minimize(
            _neg_score, pts.flatten(),
            method='L-BFGS-B',
            bounds=[(-2.5, 2.5)] * (2 * N),
            options={'maxiter': max_iter, 'maxfun': 8000, 'ftol': 1e-13},
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts


def _nm_full(pts, max_iter=800):
    """Full 26-D Nelder-Mead polish; returns improved pts."""
    try:
        res = minimize(
            _neg_score, pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-11, 'fatol': 1e-13,
                     'adaptive': True},
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts


def construct():
    t0 = time.time()
    rng = np.random.RandomState(2024)
    best_pts = None
    best_score = -1e18

    def _update(pts):
        nonlocal best_pts, best_score
        if pts is None or pts.shape != (N, 2):
            return
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    def _time_ok(margin):
        return (time.time() - t0) < (DEADLINE - margin)

    # Fallback: regular 13-gon, normalized to unit hull area.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        a0 = ConvexHull(x0).volume
        if a0 > 1e-12:
            x0 = x0 * np.sqrt(1.0 / a0)
    except Exception:
        pass
    _update(x0)

    # -----------------------------------------------------------------------
    # Phase 1: ring-space search across many symmetric structures.
    # DE (global) + L-BFGS-B (local) + NM (polish) all in the low-dim space.
    # -----------------------------------------------------------------------
    ring_structures = [
        [13], [1, 12], [1, 6, 6], [7, 6], [2, 11], [3, 10],
        [4, 9], [5, 8], [6, 7], [2, 5, 6], [3, 4, 6], [1, 5, 7],
    ]

    for rc in ring_structures:
        if not _time_ok(30):
            break

        # Global DE in ring space
        if _time_ok(30):
            p, s = _de_ring(rc, rng, maxiter=15, popsize=9)
            _update(p)

        # Local L-BFGS-B from diverse starts
        if _time_ok(30):
            p, s = _optimize_ring(rc, rng, max_iter=180, n_tries=6)
            _update(p)

        # NM polish on the best found so far, seeded by this structure's best
        if _time_ok(25) and best_pts is not None:
            p, s = _nm_ring(rc, best_pts, rng, max_iter=300)
            _update(p)

    # -----------------------------------------------------------------------
    # Phase 2: full 26-D refinement from the best ring-built configuration.
    # This breaks the ring symmetry and is where the final gain comes from.
    # -----------------------------------------------------------------------
    if best_pts is not None:
        # L-BFGS-B polish
        if _time_ok(15):
            _update(_full_polish(best_pts, rng, max_iter=350))
        # Nelder-Mead polish
        if _time_ok(12):
            _update(_nm_full(best_pts, max_iter=700))

        # A couple of perturbed full-space restarts to escape the ring basin
        for _ in range(3):
            if not _time_ok(10):
                break
            pert = best_pts.flatten() + rng.normal(0, 0.06, size=2 * N)
            try:
                res = minimize(
                    _neg_score, pert,
                    method='L-BFGS-B',
                    bounds=[(-2.5, 2.5)] * (2 * N),
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-13},
                )
                _update(res.x.reshape(N, 2))
            except Exception:
                pass
        # Final NM on the very best
        if _time_ok(6):
            _update(_nm_full(best_pts, max_iter=500))

    if best_pts is None:
        return x0
    return best_pts
