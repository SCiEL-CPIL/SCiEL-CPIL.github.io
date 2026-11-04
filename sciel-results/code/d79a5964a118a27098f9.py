"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
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
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -50, 0)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize -soft_min(ratios).
    params: 22 values representing p_3 ... p_13.
    p_1 is fixed at (0,0), p_2 is fixed at (1,0).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _soft_min(ratios, tau)
    return -approx_min

def _generate_grid_starts(rng, n_grid=3):
    """
    Generate initial points from a structured grid approach.
    For small N like 13, a grid-based initialization can be structurally superior.
    """
    starts = []
    # Create a grid of points
    grid_x = np.linspace(-1, 1, n_grid)
    grid_y = np.linspace(-1, 1, n_grid)
    
    # Create all combinations of grid points
    all_points = []
    for x in grid_x:
        for y in grid_y:
            all_points.append([x, y])
    
    all_points = np.array(all_points)
    
    # Sample N points from the grid with some perturbation
    for _ in range(10):
        idx = rng.choice(len(all_points), size=N, replace=False)
        pts = all_points[idx].copy()
        pts += rng.normal(0, 0.1, (N, 2))
        
        # Transform: p[0] -> (0,0), p[1] -> (1,0)
        if np.linalg.norm(pts[1] - pts[0]) < 1e-9:
            continue
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        
        if len(pts) == N:
            starts.append(pts[2:].flatten())
    
    return starts

def _generate_polygon_starts(rng, n_starts=10):
    """
    Generate initial points from regular polygon with perturbations.
    """
    starts = []
    for _ in range(n_starts):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles += rng.uniform(-0.15, 0.15, N)
        radii = 1.0 + rng.normal(0, 0.08, N)
        
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Transform: p[0] -> (0,0), p[1] -> (1,0)
        if np.linalg.norm(pts[1] - pts[0]) < 1e-9:
            continue
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        
        if len(pts) == N:
            starts.append(pts[2:].flatten())
    
    return starts

def _generate_halton_starts(rng, n_starts=10):
    """
    Generate initial points from Halton sequence with perturbations.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    starts = []
    for _ in range(n_starts):
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        
        # Add small random perturbation
        pts += rng.normal(0, 0.05, (N, 2))
        
        # Transform: p[0] -> (0,0), p[1] -> (1,0)
        if np.linalg.norm(pts[1] - pts[0]) < 1e-9:
            continue
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        
        if len(pts) == N:
            starts.append(pts[2:].flatten())
    
    return starts

def _params_to_points(params):
    """Convert parameter vector to point array."""
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _points_to_params(pts):
    """Convert point array to parameter vector."""
    return pts[2:].flatten()

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points using multiple methods
    starts = []
    
    # Grid-based starts (structurally superior for small N)
    grid_starts = _generate_grid_starts(rng, n_grid=3)
    starts.extend(grid_starts)
    
    # Polygon-based starts
    polygon_starts = _generate_polygon_starts(rng, n_starts=8)
    starts.extend(polygon_starts)
    
    # Halton-based starts
    halton_starts = _generate_halton_starts(rng, n_starts=8)
    starts.extend(halton_starts)
    
    # Limit total starts to manage time
    max_starts = 20
    if len(starts) > max_starts:
        # Keep best scoring starts
        start_scores = []
        for p in starts:
            pts = _params_to_points(p)
            score = _compute_min_ratio(pts)
            start_scores.append(score)
        
        sorted_indices = np.argsort(start_scores)[::-1]
        starts = [starts[idx] for idx in sorted_indices[:max_starts]]
    
    # Phase 2: Multi-scale tau annealing with L-BFGS-B
    tau_schedule = [10, 25, 50, 100, 200, 400, 800, 1500, 3000, 5000, 8000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            try:
                # Adaptive maxiter based on remaining time
                remaining = time_limit - (time.time() - start_time)
                maxiter = min(500, max(80, int(remaining * 4)))
                
                res = minimize(
                    _objective,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                # Check if improvement
                old_obj = _objective(starts[i], tau)
                new_obj = res.fun
                if new_obj < old_obj:
                    starts[i] = res.x.copy()
                
                # Evaluate true score
                pts = _params_to_points(starts[i])
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = starts[i].copy()
                    
            except Exception:
                pass
    
    # Phase 3: Differential evolution-inspired global search
    if (time.time() - start_time) < time_limit - 30.0:
        # Create population from top starts
        start_scores = []
        for p in starts:
            pts = _params_to_points(p)
            start_scores.append(_compute_min_ratio(pts))
        
        # Take top 12
        sorted_indices = np.argsort(start_scores)[::-1]
        pop_size = min(12, len(starts))
        population = [starts[idx].copy() for idx in sorted_indices[:pop_size]]
        scores = [start_scores[idx] for idx in sorted_indices[:pop_size]]
        
        # DE iterations
        max_de_iters = 80
        for it in range(max_de_iters):
            if (time.time() - start_time) > time_limit - 15.0:
                break
            
            # DE mutation
            new_idx = rng.integers(0, pop_size)
            candidates = [i for i in range(pop_size) if i != new_idx]
            if len(candidates) < 3:
                break
            a = rng.choice(candidates)
            b = rng.choice([i for i in candidates if i != a])
            c = rng.choice([i for i in candidates if i != a and i != b])
            
            F = rng.uniform(0.3, 0.9)
            trial = population[a] + F * (population[b] - population[c])
            
            # Evaluate
            pts = _params_to_points(trial)
            score = _compute_min_ratio(pts)
            
            if score > scores[new_idx]:
                population[new_idx] = trial.copy()
                scores[new_idx] = score
                
                # Local polish on improved point
                if (time.time() - start_time) < time_limit - 20.0:
                    try:
                        res = minimize(
                            _objective,
                            trial,
                            args=(500,),
                            method='L-BFGS-B',
                            options={'maxiter': 150, 'ftol': 1e-14}
                        )
                        if res.fun < _objective(trial, 500):
                            polished = res.x.copy()
                            pts2 = _params_to_points(polished)
                            score2 = _compute_min_ratio(pts2)
                            if score2 > scores[new_idx]:
                                population[new_idx] = polished
                                scores[new_idx] = score2
                                if score2 > best_score:
                                    best_score = score2
                                    best_params = polished.copy()
                    except Exception:
                        pass
    
    # Phase 4: Final high-precision polish on the best solution
    if best_params is not None and (time.time() - start_time) < time_limit - 8.0:
        # Multiple refinement passes with increasing tau
        for tau in [1000, 2000, 4000, 8000, 15000, 25000]:
            if (time.time() - start_time) > time_limit - 5.0:
                break
            try:
                res = minimize(
                    _objective,
                    best_params,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if res.fun < _objective(best_params, tau):
                    best_params = res.x.copy()
                    pts = _params_to_points(best_params)
                    score = _compute_min_ratio(pts)
                    if score > best_score:
                        best_score = score
            except Exception:
                pass
        
        # Final Nelder-Mead polish for very fine tuning
        if (time.time() - start_time) < time_limit - 3.0:
            try:
                def nm_objective(params):
                    pts = _params_to_points(params)
                    return -_compute_min_ratio(pts)
                
                res_nm = minimize(
                    nm_objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_nm.fun < nm_objective(best_params):
                    best_params = res_nm.x.copy()
                    pts = _params_to_points(best_params)
                    score = _compute_min_ratio(pts)
                    if score > best_score:
                        best_score = score
            except Exception:
                pass
    
    # Construct final points
    if best_params is None:
        best_params = _generate_polygon_starts(rng, 1)[0]
    
    final_pts = _params_to_points(best_params)
    
    return final_pts

# EVOLVE-BLOCK-END

