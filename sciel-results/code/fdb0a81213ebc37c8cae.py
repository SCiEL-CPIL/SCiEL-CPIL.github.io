"""Optimized program: convex polygon parameterization + soft-min + DE + Powell polish."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14
# Precompute all triangle index combinations
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_NUM_TRIS = _TRI_INDICES.shape[0]  # 364

def _triangle_areas(points):
    """Vectorized triangle areas for all C(N,3) triples. points: (N,2)"""
    i, j, k = _TRI_INDICES[:, 0], _TRI_INDICES[:, 1], _TRI_INDICES[:, 2]
    v1 = points[j] - points[i]
    v2 = points[k] - points[i]
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(points):
    """Convex hull area, 0 if degenerate."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """min triangle area / hull area"""
    areas = _triangle_areas(points)
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    return float(np.min(areas)) / ha

def _soft_min(areas, beta):
    """Soft minimum via log-sum-exp (negative for min)."""
    # min(a) ≈ -1/beta * log(sum(exp(-beta * a)))
    m = np.min(areas)
    return m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta

def _param_to_points(params):
    """
    params: (2N,) = [r0, r1, ..., r13, dtheta0, dtheta1, ..., dtheta13]
    Returns (N,2) points as convex polygon.
    Base: regular 14-gon on unit circle.
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = params[:N]
    dthetas = params[N:]
    angles = base_angles + dthetas
    # Sort by angle to maintain convex polygon ordering
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted = radii[order]
    x = radii_sorted * np.cos(angles_sorted)
    y = radii_sorted * np.sin(angles_sorted)
    return np.column_stack([x, y])

def _param_score(params, beta):
    """Objective for parameter-space optimization (negative soft score)."""
    points = _param_to_points(params)
    areas = _triangle_areas(points)
    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e6
    soft = _soft_min(areas, beta)
    return -soft / ha

def _cart_score_neg(points_flat):
    """Negative exact score for Cartesian Powell."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _regular_14gon():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    # --- Phase 0: Initial candidates ---
    base = _regular_14gon()
    best_points = base.copy()
    best_score = _score(base)

    # Random perturbations of regular polygon
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        pert = base + 0.15 * rng.standard_normal((N, 2))
        s = _score(pert)
        if s > best_score:
            best_score = s
            best_points = pert.copy()

    # --- Phase 1: Differential Evolution in parameter space ---
    # params: radii in [0.3, 1.5], dthetas in [-0.4, 0.4]
    bounds = [(0.3, 1.5)] * N + [(-0.4, 0.4)] * N
    beta_start = 20.0
    beta_end = 200.0
    n_iter_de = 80

    if time.time() < deadline - 40:
        # Adaptive beta: start low (smoother), increase for sharpness
        beta_schedule = np.linspace(beta_start, beta_end, n_iter_de)
        
        def de_objective(params):
            # Use current beta based on iteration (approximate via global counter)
            return _param_score(params, beta_current[0])
        
        beta_current = [beta_start]
        de_iter = [0]
        
        def de_obj_wrapper(params):
            de_iter[0] += 1
            frac = min(de_iter[0] / n_iter_de, 1.0)
            beta_current[0] = beta_start + (beta_end - beta_start) * frac
            return _param_score(params, beta_current[0])

        try:
            de_res = differential_evolution(
                de_obj_wrapper,
                bounds,
                maxiter=n_iter_de,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False,
                init='sobol',
                workers=1,
                updating='immediate',
                callback=lambda p, conv: (time.time() > deadline - 30)
            )
            if de_res.fun < 1e5:
                p_opt = _param_to_points(de_res.x)
                s_opt = _score(p_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = p_opt.copy()
        except Exception:
            pass

    # --- Phase 2: Powell refinement in Cartesian space ---
    # Multiple restarts from best + perturbations
    n_powell_restarts = 6
    for r in range(n_powell_restarts):
        if time.time() > deadline - 15:
            break
        
        if r == 0:
            x0 = best_points.flatten()
        else:
            x0 = best_points.flatten() + 0.05 * rng.standard_normal(2 * N)
        
        try:
            res = minimize(
                _cart_score_neg, x0,
                method='Powell',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 5000}
            )
            if res.fun < -best_score:
                p_opt = res.x.reshape(-1, 2)
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = p_opt.copy()
        except Exception:
            pass

    # --- Phase 3: Final Nelder-Mead polish on best ---
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _cart_score_neg, best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if res.fun < -best_score:
                p_opt = res.x.reshape(-1, 2)
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = p_opt.copy()
        except Exception:
            pass

    return best_points
