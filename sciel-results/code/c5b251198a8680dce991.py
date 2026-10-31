"""
Nelder-Mead multi-start + differential evolution hybrid.
Uses derivative-free local search (Nelder-Mead) on the non-smooth min-area objective,
combined with DE for global exploration. Different optimization paradigm from SA+gradient.
"""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]


def _clamp_all(pts):
    """Vectorized clamp to triangle."""
    pts = np.array(pts, dtype=float).copy()
    x = pts[:, 0]
    y = pts[:, 1]
    y = np.clip(y, _EPS, _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = np.clip(x, x_min, x_max)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _areas_vec(pts):
    """All 165 triangle areas (vectorized)."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _min_area(pts):
    return float(np.min(_areas_vec(pts)))


def _objective_exact(params):
    """Exact non-smooth objective: negative min area."""
    pts = _clamp_all(params.reshape(N, 2))
    return -_min_area(pts)


def _nelder_mead_refine(pts, time_limit=15.0, maxiter=500):
    """
    Nelder-Mead local search on the non-smooth min-area objective.
    """
    x0 = _clamp_all(np.array(pts, dtype=float)).flatten()
    start_time = time.time()
    
    best_x = x0.copy()
    best_score = _min_area(best_x.reshape(N, 2))
    
    try:
        res = minimize(
            _objective_exact,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            },
            callback=lambda x: None,
        )
        cand = _clamp_all(res.x.reshape(N, 2))
        s = _min_area(cand)
        if s > best_score:
            best_score = s
            best_x = cand.flatten()
    except Exception:
        pass
    
    # If time remains, do another NM pass from best
    if time.time() - start_time < time_limit * 0.7:
        try:
            res2 = minimize(
                _objective_exact,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': maxiter // 2,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True,
                },
            )
            cand2 = _clamp_all(res2.x.reshape(N, 2))
            s2 = _min_area(cand2)
            if s2 > best_score:
                best_score = s2
                best_x = cand2.flatten()
        except Exception:
            pass
    
    return _clamp_all(best_x.reshape(N, 2)), best_score


def _de_global(bounds, popsize=10, maxiter=30, tol=1e-8, time_limit=40.0, seed=42):
    """
    Differential evolution for global exploration.
    """
    start_time = time.time()
    
    def callback(xk, convergence):
        return time.time() - start_time > time_limit
    
    try:
        res = differential_evolution(
            _objective_exact,
            bounds,
            popsize=popsize,
            maxiter=maxiter,
            tol=tol,
            mutation=(0.3, 0.9),
            recombination=0.7,
            seed=seed,
            callback=callback,
            polish=True,
        )
        cand = _clamp_all(res.x.reshape(N, 2))
        s = _min_area(cand)
        return cand, s
    except Exception:
        return None, -1.0


def _generate_seeds():
    """Generate diverse initial configurations."""
    h = _H
    seeds = []
    
    # Seed 1: Vertices + centroid + hexagonal pattern
    s1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.167, h/12],
        [0.833, h/12],
        [0.333, h/4],
        [0.667, h/4],
    ])
    seeds.append(_clamp_all(s1))
    
    # Seed 2: Triangular grid
    s2 = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, h/4],
        [0.5, h/4],
        [0.75, h/4],
        [0.5, h/2],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, 2*h/3],
        [0.25, 2*h/3],
    ])
    seeds.append(_clamp_all(s2))
    
    # Seed 3: 3-fold symmetric
    s3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.167, h/6],
        [0.833, h/6],
        [0.5, h/2],
        [0.333, 2*h/3],
        [0.667, 2*h/3],
        [0.25, h/4],
        [0.75, h/4],
    ])
    seeds.append(_clamp_all(s3))
    
    # Seed 4: Random
    rng = np.random.default_rng(123)
    for _ in range(8):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        seeds.append(_clamp_all(pts))
        
    # Seed 5: Near edges
    s5 = np.array([
        [0.1, 0.05],
        [0.9, 0.05],
        [0.5, h - 0.05],
        [0.5, h/3],
        [0.3, h/6],
        [0.7, h/6],
        [0.5, h/2],
        [0.2, h/4],
        [0.8, h/4],
        [0.4, 2*h/5],
        [0.6, 2*h/5],
    ])
    seeds.append(_clamp_all(s5))
    
    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_pts = None
    best_score = -1.0
    
    def _update(pts, score):
        nonlocal best_pts, best_score
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()
    
    # Phase 1: Nelder-Mead from multiple seeds (fast local search)
    seeds = _generate_seeds()
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 60:
            break
        remaining = deadline - 60 - time.time()
        if remaining < 2:
            break
        per = min(5.0, remaining / max(1, len(seeds) - i))
        if per < 1.5:
            break
        try:
            nm_pts, nm_score = _nelder_mead_refine(seed, time_limit=per, maxiter=300)
            _update(nm_pts, nm_score)
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = seeds[0]
        best_score = _min_area(best_pts)
    
    # Phase 2: Differential evolution for global exploration
    if time.time() < deadline - 30:
        remaining = deadline - 30 - time.time()
        if remaining > 20:
            # Bounds for each point in triangle
            bounds = [(0.0, 1.0), (0.0, _H)] * N
            try:
                de_pts, de_score = _de_global(
                    bounds, popsize=8, maxiter=25,
                    time_limit=min(35.0, remaining), seed=42
                )
                if de_pts is not None:
                    _update(de_pts, de_score)
            except Exception:
                pass
    
    # Phase 3: NM polish from best
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - 15 - time.time()
        if remaining > 3:
            try:
                nm2_pts, nm2_score = _nelder_mead_refine(
                    best_pts, time_limit=min(12.0, remaining), maxiter=400
                )
                _update(nm2_pts, nm2_score)
            except Exception:
                pass
    
    # Phase 4: Final tight NM
    if best_pts is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        if remaining > 2:
            try:
                nm3_pts, nm3_score = _nelder_mead_refine(
                    best_pts, time_limit=min(5.0, remaining), maxiter=200
                )
                _update(nm3_pts, nm3_score)
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = seeds[0]
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
