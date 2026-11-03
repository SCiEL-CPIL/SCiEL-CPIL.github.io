"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

# Precompute all triangle indices for N=13: C(13,3)=286
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _min_triangle_area_raw(pts):
    """Compute minimum triangle area for n points (vectorized)."""
    a = pts[_TRIPLETS[:, 0]]
    b = pts[_TRIPLETS[:, 1]]
    c = pts[_TRIPLETS[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_ta = _min_triangle_area_raw(pts)
    return min_ta / hull_a

def _soft_min_area(pts, beta):
    """Soft-min surrogate for minimum triangle area (vectorized)."""
    a = pts[_TRIPLETS[:, 0]]
    b = pts[_TRIPLETS[:, 1]]
    c = pts[_TRIPLETS[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    m = np.min(areas)
    if m < 1e-15:
        return m
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    return m - (1.0 / beta) * log_sum

def _objective(x):
    """Objective to minimize: negative score (vectorized, fast)."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _soft_objective_factory(beta):
    """Factory for soft-min objective used in L-BFGS-B polishing."""
    def obj(x):
        pts = x.reshape(N, 2)
        return -_soft_min_area(pts, beta)
    return obj

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def _center_ring(n_ring, inner_r, outer_r):
    """One center point + points on two concentric rings."""
    pts = [[0.5, 0.5]]
    n1 = n_ring // 2
    n2 = n_ring - n1
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    pts.extend(np.column_stack([0.5 + outer_r * np.cos(a1), 0.5 + outer_r * np.sin(a1)]).tolist())
    pts.extend(np.column_stack([0.5 + inner_r * np.cos(a2), 0.5 + inner_r * np.sin(a2)]).tolist())
    return np.array(pts[:N])

def _polish(pts, deadline, max_beta=2000):
    """Polish a point set using L-BFGS-B with soft-min objective."""
    current = pts.copy().flatten()
    best_pts = pts.copy()
    best_score = _score(pts)
    for beta in [100, 500, max_beta]:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _soft_objective_factory(beta),
                current,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            current = result.x
            cand = current.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Build diverse structured starting points
    starts = [
        _regular_polygon(N, radius=0.5, center=(0.5, 0.5)),
        _regular_polygon(N, radius=0.45, center=(0.5, 0.5)),
        _regular_polygon(N, radius=0.55, center=(0.5, 0.5)),
        _center_ring(12, inner_r=0.25, outer_r=0.45),
        _center_ring(12, inner_r=0.30, outer_r=0.40),
        np.array([[0.5, 0.5]] + list(_regular_polygon(12, radius=0.5, center=(0.5, 0.5)))),
    ]

    best_pts = starts[0].copy()
    best_score = _score(best_pts)
    for s in starts[1:]:
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_pts = s.copy()

    bounds = [(0.0, 1.0)] * (N * 2)

    # Global search: Differential Evolution (vectorized objective)
    de_configs = [
        dict(maxiter=120, popsize=15, seed=42, tol=1e-8, polish=False),
        dict(maxiter=120, popsize=15, seed=123, tol=1e-8, polish=False),
        dict(maxiter=120, popsize=20, seed=777, tol=1e-8, polish=False),
    ]

    for cfg in de_configs:
        if time.time() > deadline - 20:
            break
        try:
            result = differential_evolution(_objective, bounds, **cfg)
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape(N, 2).copy()
        except Exception:
            pass

    # Polish the best point found with L-BFGS-B soft-min
    if time.time() < deadline - 5:
        polished, polished_score = _polish(best_pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished

    # Polish a couple of structured starts too (cheap, may beat DE)
    if time.time() < deadline - 5:
        for s in starts[:3]:
            if time.time() > deadline - 5:
                break
            try:
                polished, polished_score = _polish(s, deadline)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished
            except Exception:
                pass

    if best_pts is None:
        best_pts = _regular_polygon(N)

    return best_pts
# EVOLVE-BLOCK-END
