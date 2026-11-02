"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull


def _triangle_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))


def _min_triangle_area(pts):
    """Minimum area over all C(n,3) triples."""
    n = len(pts)
    best = float('inf')
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pq = pts[j] - pi
            for k in range(j + 1, n):
                pr = pts[k] - pi
                a = 0.5 * abs(pq[0] * pr[1] - pq[1] * pr[0])
                if a < best:
                    best = a
    return best


def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12


def _score(pts):
    m = _min_triangle_area(pts)
    h = _hull_area(pts)
    return m / h if h > 1e-15 else 0.0


def _ring_init(n=14, k=7, r=1.0, phase=0.0):
    """k inner points at radius r, plus outer ring to reach n points, all convex-ish."""
    pts = []
    # inner regular k-gon
    for i in range(k):
        th = phase + 2 * np.pi * i / k
        pts.append([r * np.cos(th), r * np.sin(th)])
    # outer points: distribute remaining n-k on unit circle
    m = n - k
    for i in range(m):
        th = phase + 2 * np.pi * (i + 0.5) / m
        pts.append([np.cos(th), np.sin(th)])
    return np.array(pts, dtype=float)


def _reflect(pts):
    """Central-symmetrize: average with -pts (keeps shape, enforces symmetry)."""
    return 0.5 * (pts - pts)  # placeholder; real sym below


def _make_symmetric(pts):
    """Map to centrally symmetric set: for each point keep both p and -p,
    then pick n points as a symmetric subset."""
    # Build symmetric candidate set from pts and -pts
    cands = np.vstack([pts, -pts])
    # We need exactly n=14 points, symmetric => 7 pairs.
    # Pick 7 directions from pts (use the 7 with largest norm? or first 7)
    # Simpler: take pts[:7] as one half, mirror the rest.
    half = pts[:7]
    return np.vstack([half, -half])


def _local_refine(pts, iters=3000, deadline=None):
    """Coordinate-descent / Nelder-Mead style local search on min-triangle ratio."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    step = 0.05
    n = len(best_pts)
    t0 = time.time()
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        # try a small random perturbation on a random point
        idx = np.random.randint(n)
        d = np.random.randn(2) * step
        trial = best_pts.copy()
        trial[idx] += d
        s = _score(trial)
        if s > best_score:
            best_pts = trial
            best_score = s
            # occasionally grow step
            step = min(step * 1.02, 0.3)
        else:
            step = max(step * 0.97, 0.005)
    return best_pts, best_score


def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    rng = np.random.default_rng(42)
    best_pts = None
    best_score = 0.0

    # ---- Phase 1: diverse ring-based initializations ----
    for trial in range(60):
        if time.time() > deadline - 20:
            break
        k = 7  # inner ring size
        r = 0.3 + 0.5 * rng.random()
        phase = rng.random() * 2 * np.pi
        pts = _ring_init(N, k=k, r=r, phase=phase)
        # random shuffle to break symmetry of ordering
        pts = pts[rng.permutation(N)]
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # ---- Phase 2: local refinement from best ----
    if best_pts is not None:
        refined, s = _local_refine(best_pts, iters=4000, deadline=deadline - 5)
        if s > best_score:
            best_pts = refined
            best_score = s

    # ---- Phase 3: a few more random restarts with refinement if time ----
    for trial in range(8):
        if time.time() > deadline - 5:
            break
        k = int(rng.integers(4, 9))
        r = 0.2 + 0.6 * rng.random()
        phase = rng.random() * 2 * np.pi
        pts = _ring_init(N, k=k, r=r, phase=phase)
        pts = pts[rng.permutation(N)]
        s = _score(pts)
        # quick refine
        if s > best_score * 0.9:
            refined, s2 = _local_refine(pts, iters=1500, deadline=deadline - 3)
            if s2 > best_score:
                best_pts = refined
                best_score = s2

    if best_pts is None:
        # fallback: regular polygon
        th = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(th), np.sin(th)])

    return best_pts
# EVOLVE-BLOCK-END
