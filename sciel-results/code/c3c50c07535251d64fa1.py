import time
import math
import numpy as np
from scipy.optimize import minimize, basinhopping

_N = 8
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _diameter(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())

def _log_product(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    # Sum of logs for i < j, multiply by 2 for ordered pairs
    tri_idx = np.triu_indices(_N, 1)
    return 2.0 * float(np.sum(np.log(d[tri_idx])))

def _score_from_log_product_and_diameter(lp, dia):
    if dia <= 1e-12:
        return 0.0
    n = _N
    return math.exp(lp + n * (n - 1) * (math.log(2.0) - math.log(dia)) - n * math.log(n))

def _score(pts):
    dia = _diameter(pts)
    if dia <= 1e-12:
        return 0.0
    return _score_from_log_product_and_diameter(_log_product(pts), dia)

def _objective(pts):
    dia = _diameter(pts)
    if dia < 1e-8:
        return 1e12
    return -_score_from_log_product_and_diameter(_log_product(pts), dia)

def _refine(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Stage 1: L-BFGS-B
    try:
        res = minimize(_objective, pts.ravel(), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        cand = res.x.reshape(_N, 2)
        s = _score(cand)
        if s > best_score:
            best_pts, best_score = cand, s
    except Exception:
        pass

    # Stage 2: SLSQP
    if _remaining() > 5.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='SLSQP', options={'maxiter': 200, 'ftol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 3: High precision L-BFGS-B
    if _remaining() > 2.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 4: Even tighter polish with BFGS (unconstrained)
    if _remaining() > 1.5:
        try:
            res = minimize(_objective, best_pts.ravel(), method='BFGS', options={'maxiter': 500, 'gtol': 1e-16})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    return best_pts, best_score

def _gaps_to_points(gaps):
    """Convert 7 angular gaps (summing to 2*pi) to 8 points on unit circle.
    The 8th gap is determined: gap_7 = 2*pi - sum(gaps[0:7])
    """
    gaps = np.asarray(gaps, dtype=float)
    total = gaps.sum()
    if total <= 1e-10:
        # Fallback to regular octagon
        thetas = 2 * math.pi * np.arange(_N) / _N
        return np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    # Compute cumulative angles starting from 0
    angles = np.zeros(_N)
    angles[0] = 0.0
    for i in range(1, _N):
        angles[i] = angles[i-1] + gaps[i-1]
    
    # Scale so that total span is 2*pi (in case gaps don't sum exactly)
    angles = angles * (2 * math.pi / angles[-1]) if angles[-1] > 1e-10 else angles
    
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _gaps_objective(gaps):
    """Objective for optimizing 7 angular gaps. Points are on unit circle."""
    gaps = np.asarray(gaps, dtype=float)
    if gaps.min() < 1e-6:
        return 1e10  # Penalize zero/negative gaps
    
    pts = _gaps_to_points(gaps)
    return _objective(pts)

def _optimize_gaps(gaps_init, max_iter=1000):
    """Optimize 7 angular gaps using SLSQP with constraints."""
    n_gaps = _N - 1  # 7 gaps
    
    # Constraints: each gap > 0, sum of gaps = 2*pi
    constraints = []
    # Sum constraint
    constraints.append({'type': 'eq', 'fun': lambda g: g.sum() - 2 * math.pi})
    # Each gap positive (use inequality: g_i - epsilon >= 0)
    eps = 1e-8
    for i in range(n_gaps):
        constraints.append({'type': 'ineq', 'fun': lambda g, i=i: g[i] - eps})
    
    bounds = [(eps, 2 * math.pi)] * n_gaps
    
    try:
        res = minimize(_gaps_objective, gaps_init, method='SLSQP', 
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        return res.x
    except Exception:
        return gaps_init

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best = None
    best_score = 0.0
    
    # 1. Try regular octagon
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    cand, s = _refine(oct_pts)
    best, best_score = cand, s

    # 2. Angular gap optimization (7D) - this is the key new approach
    if _remaining() > 15.0:
        # Start from regular octagon gaps
        regular_gap = 2 * math.pi / _N
        gaps_init = np.full(_N - 1, regular_gap)
        
        # Try multiple perturbations of the regular gaps
        gap_perturbations = [
            np.full(_N - 1, regular_gap),  # Regular
            np.array([regular_gap * 1.2, regular_gap * 0.8, regular_gap, regular_gap,
                      regular_gap, regular_gap, regular_gap]),
            np.array([regular_gap * 0.8, regular_gap * 1.2, regular_gap, regular_gap,
                      regular_gap, regular_gap, regular_gap]),
            np.array([regular_gap * 1.5, regular_gap * 0.5, regular_gap, regular_gap,
                      regular_gap, regular_gap, regular_gap]),
            np.array([regular_gap, regular_gap, regular_gap * 1.5, regular_gap * 0.5,
                      regular_gap, regular_gap, regular_gap]),
            np.array([regular_gap * 1.1, regular_gap * 0.9, regular_gap * 1.1, regular_gap * 0.9,
                      regular_gap * 1.1, regular_gap * 0.9, regular_gap]),
        ]
        
        for gaps_init in gap_perturbations:
            if _remaining() < 10.0:
                break
            gaps_opt = _optimize_gaps(gaps_init, max_iter=500)
            pts_gaps = _gaps_to_points(gaps_opt)
            cand, s = _refine(pts_gaps)
            if s > best_score:
                best, best_score = cand, s
        
        # Random gap perturbations
        for _ in range(5):
            if _remaining() < 10.0:
                break
            gaps_rand = rng.uniform(0.1, 2 * math.pi, size=_N - 1)
            gaps_rand = gaps_rand / gaps_rand.sum() * 2 * math.pi
            gaps_opt = _optimize_gaps(gaps_rand, max_iter=300)
            pts_gaps = _gaps_to_points(gaps_opt)
            cand, s = _refine(pts_gaps)
            if s > best_score:
                best, best_score = cand, s

    # 3. Radial deformations
    if _remaining() > 10.0:
        def radial_objective(radii):
            radii = np.asarray(radii, dtype=float)
            thetas = 2 * math.pi * np.arange(_N) / _N
            pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
            return _objective(pts)

        r0 = np.ones(_N)
        bounds = [(0.1, 2.0)] * _N
        try:
            res_r = minimize(radial_objective, r0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            radii_opt = res_r.x
            thetas = 2 * math.pi * np.arange(_N) / _N
            pts_radial = np.column_stack([radii_opt * np.cos(thetas), radii_opt * np.sin(thetas)])
            cand, s = _refine(pts_radial)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 4. Basin Hopping on full 16D space
    if _remaining() > 20.0:
        def bh_objective(x):
            return _objective(x.reshape(_N, 2))
        
        try:
            bh_res = basinhopping(
                bh_objective, best.ravel(), 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-10}},
                niter=80, seed=12345, 
                stepsize=0.5,
                T=100.0,
                accept_test=lambda x, fx, x0, fx0: fx < fx0
            )
            cand = bh_res.x.reshape(_N, 2)
            cand, s = _refine(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 5. Two-squares configuration
    if _remaining() > 10.0:
        pts_butterfly = []
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts_butterfly.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts_butterfly.append([0.5 * math.cos(ang), 0.5 * math.sin(ang)])
        pts_butterfly = np.array(pts_butterfly, dtype=float)
        cand, s = _refine(pts_butterfly)
        if s > best_score:
            best, best_score = cand, s

    # 6. Random restarts
    while _remaining() > 10.0:
        pts_rand = rng.uniform(-1.0, 1.0, size=(8, 2))
        cand, s = _refine(pts_rand)
        if s > best_score:
            best, best_score = cand, s

    # Final high-precision polish
    if _remaining() > 2.0:
        cand, s = _refine(best)
        if s > best_score:
            best, best_score = cand, s

    best = np.asarray(best, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points
    for i in range(_N):
        for j in range(i+1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-5:
                best[j] += [1e-5, 0]
                
    return [[float(x), float(y)] for (x, y) in best]
