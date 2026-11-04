"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))
_TRI_I = _triple_indices[:, 0]
_TRI_J = _triple_indices[:, 1]
_TRI_K = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-12:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -500, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective_full(params, tau):
    """
    Objective: minimize -smooth_min(triangle_area / hull_area).
    params: 26 values (13 points × 2 coords).
    """
    pts = params.reshape(N, 2)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _make_regular_polygon(rng, n=N, noise=0.0):
    """Generate a regular n-gon, optionally with noise."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if noise > 0:
        angles += rng.uniform(-noise, noise, n)
        radii = 1.0 + rng.normal(0, noise * 0.5, n)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    else:
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
    return pts.flatten()

def _make_halton(rng, n=N):
    """Halton sequence in 2D."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 2))
    for i in range(n):
        pts[i, 0] = halton(i + 1, 2)
        pts[i, 1] = halton(i + 1, 3)
    pts += rng.normal(0, 0.03, (n, 2))
    return pts.flatten()

def _make_spiral(rng, n=N):
    """Golden-ratio spiral."""
    pts = np.zeros((n, 2))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = 0.15 * np.sqrt(i + 1)
        theta = i * golden_angle
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
    pts += rng.normal(0, 0.05, (n, 2))
    return pts.flatten()

def _make_random(rng, n=N):
    """Random points in unit square, centered."""
    pts = rng.uniform(-1, 1, (n, 2))
    return pts.flatten()

def _make_concentric(rng, n=N):
    """Concentric rings: inner 5, outer 8."""
    pts = np.zeros((n, 2))
    # Inner ring: 5 points
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rng.uniform(-0.2, 0.2, 5)
    inner_radii = 0.4 + rng.normal(0, 0.03, 5)
    pts[:5, 0] = inner_radii * np.cos(inner_angles)
    pts[:5, 1] = inner_radii * np.sin(inner_angles)
    # Outer ring: 8 points
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + rng.uniform(-0.15, 0.15, 8)
    outer_radii = 1.0 + rng.normal(0, 0.03, 8)
    pts[5:, 0] = outer_radii * np.cos(outer_angles)
    pts[5:, 1] = outer_radii * np.sin(outer_angles)
    return pts.flatten()

def _nelder_mead_polish(params, time_budget=5.0):
    """Nelder-Mead polish on the exact nonsmooth objective."""
    def exact_obj(p):
        pts = p.reshape(N, 2)
        score = _compute_min_ratio(pts)
        return -score
    
    t0 = time.time()
    try:
        res = minimize(
            exact_obj,
            params,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True}
        )
        elapsed = time.time() - t0
        if elapsed > time_budget:
            return params
        new_pts = res.x.reshape(N, 2)
        new_score = _compute_min_ratio(new_pts)
        old_score = -exact_obj(params)
        if new_score > old_score:
            return res.x.copy()
        return params
    except Exception:
        return params

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    starts = []
    start_generators = [
        lambda: _make_regular_polygon(rng, noise=0.0),
        lambda: _make_regular_polygon(rng, noise=0.05),
        lambda: _make_regular_polygon(rng, noise=0.15),
        lambda: _make_halton(rng),
        lambda: _make_spiral(rng),
        lambda: _make_random(rng),
        lambda: _make_concentric(rng),
        lambda: _make_regular_polygon(rng, noise=0.3),
        lambda: _make_halton(rng),
        lambda: _make_spiral(rng),
        lambda: _make_random(rng),
        lambda: _make_concentric(rng),
    ]
    
    for gen in start_generators:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        try:
            p = gen()
            starts.append(p)
        except Exception:
            pass
    
    # Phase 2: Multi-tau L-BFGS-B optimization
    tau_schedule = [10, 30, 80, 200, 500, 1200, 3000, 8000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 10.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            try:
                res = minimize(
                    _objective_full,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13,
                             'maxfun': 10000}
                )
                
                new_params = res.x
                pts = new_params.reshape(N, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = new_params.copy()
                
                starts[i] = new_params.copy()
                
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish on the best solution
    if best_params is not None and (time.time() - start_time) < time_limit - 20.0:
        polished = _nelder_mead_polish(best_params, time_budget=10.0)
        pts = polished.reshape(N, 2)
        score = _compute_min_ratio(pts)
        if score > best_score:
            best_score = score
            best_params = polished.copy()
    
    # Phase 4: Perturbation + re-optimize around the best
    if (time.time() - start_time) < time_limit - 15.0:
        num_perturb = 8
        for k in range(num_perturb):
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            sigma = 0.1 / (k + 1)
            perturbed = best_params + rng.normal(0, sigma, 2 * N)
            
            try:
                res = minimize(
                    _objective_full,
                    perturbed,
                    args=(500,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts = res.x.reshape(N, 2)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
            
            if (time.time() - start_time) < time_limit - 12.0:
                polished = _nelder_mead_polish(best_params, time_budget=5.0)
                pts = polished.reshape(N, 2)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_params = polished.copy()
    
    # Phase 5: Final high-precision L-BFGS-B + NM polish
    if best_params is not None and (time.time() - start_time) < time_limit - 10.0:
        try:
            res = minimize(
                _objective_full,
                best_params,
                args=(15000,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14,
                         'maxfun': 20000}
            )
            pts = res.x.reshape(N, 2)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final NM polish
    if best_params is not None and (time.time() - start_time) < time_limit - 3.0:
        polished = _nelder_mead_polish(best_params, time_budget=8.0)
        pts = polished.reshape(N, 2)
        score = _compute_min_ratio(pts)
        if score > best_score:
            best_score = score
            best_params = polished.copy()
    
    # Reconstruct final points
    if best_params is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_params.reshape(N, 2)

# EVOLVE-BLOCK-END
