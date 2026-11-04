"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
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

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    More numerically stable version.
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    # Clip to avoid overflow (shifted should be <= 0)
    shifted = np.clip(shifted, -50, 0)
    # Use logsumexp for numerical stability
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(ratios).
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
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _generate_initial_params(rng, method='polygon'):
    """
    Generate initial parameters for the 22 free coordinates.
    """
    if method == 'polygon':
        # Regular 13-gon with random perturbation
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles += rng.uniform(-0.15, 0.15, N)
        radii = 1.0 + rng.normal(0, 0.08, N)
        
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Transform: p[0] -> (0,0), p[1] -> (1,0)
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
    
    elif method == 'random':
        return rng.uniform(-1, 1, 22)
    
    elif method == 'halton':
        # Halton sequence for low-discrepancy
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        
        # Add small random perturbation
        pts += rng.normal(0, 0.05, (N, 2))
        
        # Transform
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
    
    else:
        return rng.uniform(-1, 1, 22)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    num_starts = 16
    starts = []
    start_methods = ['polygon', 'halton', 'random', 'polygon', 'halton', 'random',
                     'polygon', 'halton', 'random', 'polygon', 'halton', 'random',
                     'polygon', 'halton', 'random', 'polygon']
    
    for i in range(num_starts):
        if (time.time() - start_time) > time_limit - 25.0:
            break
        method = start_methods[i % len(start_methods)]
        p = _generate_initial_params(rng, method)
        starts.append(p)
    
    # Phase 2: Local optimization with L-BFGS-B using smooth min
    # Use multiple tau values for annealing - more aggressive schedule
    tau_schedule = [15, 40, 100, 250, 600]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 8.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                # Check if improvement
                old_obj = _objective(starts[i], tau)
                new_obj = res.fun
                if new_obj < old_obj:
                    starts[i] = res.x.copy()
                
                # Evaluate true score
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = starts[i].reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = starts[i].copy()
                    
            except Exception:
                pass
    
    # Phase 3: DE-inspired global search with local polish
    if (time.time() - start_time) < time_limit - 20.0:
        # Create larger population from top starts
        # Sort starts by their current score
        start_scores = []
        for p in starts:
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = p.reshape(N - 2, 2)
            start_scores.append(_compute_min_ratio(pts))
        
        # Take top 8
        sorted_indices = np.argsort(start_scores)[::-1]
        population = [starts[idx].copy() for idx in sorted_indices[:8]]
        scores = [start_scores[idx] for idx in sorted_indices[:8]]
        
        # DE iterations
        max_de_iters = 40
        for it in range(max_de_iters):
            if (time.time() - start_time) > time_limit - 8.0:
                break
            
            # DE mutation
            pop_size = len(population)
            new_idx = rng.integers(0, pop_size)
            candidates = [i for i in range(pop_size) if i != new_idx]
            a = rng.choice(candidates)
            b = rng.choice([i for i in candidates if i != a])
            c = rng.choice([i for i in candidates if i != a and i != b])
            
            F = rng.uniform(0.4, 1.0)
            trial = population[a] + F * (population[b] - population[c])
            
            # Evaluate
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = trial.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            
            if score > scores[new_idx]:
                population[new_idx] = trial.copy()
                scores[new_idx] = score
                
                # Local polish on improved point
                if (time.time() - start_time) < time_limit - 12.0:
                    try:
                        res = minimize(
                            _objective,
                            trial,
                            args=(300,),
                            method='L-BFGS-B',
                            options={'maxiter': 150, 'ftol': 1e-14}
                        )
                        if res.fun < _objective(trial, 300):
                            polished = res.x.copy()
                            pts2 = np.empty((N, 2))
                            pts2[0] = [0.0, 0.0]
                            pts2[1] = [1.0, 0.0]
                            pts2[2:] = polished.reshape(N - 2, 2)
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
    if best_params is not None and (time.time() - start_time) < time_limit - 5.0:
        # Multiple refinement passes with increasing tau
        for tau in [800, 1500, 3000, 5000]:
            if (time.time() - start_time) > time_limit - 3.0:
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
                    pts = np.empty((N, 2))
                    pts[0] = [0.0, 0.0]
                    pts[1] = [1.0, 0.0]
                    pts[2:] = best_params.reshape(N - 2, 2)
                    score = _compute_min_ratio(pts)
                    if score > best_score:
                        best_score = score
            except Exception:
                pass
        
        # Final Nelder-Mead polish for very fine tuning
        if (time.time() - start_time) < time_limit - 2.0:
            try:
                def nm_objective(params):
                    pts = np.empty((N, 2))
                    pts[0] = [0.0, 0.0]
                    pts[1] = [1.0, 0.0]
                    pts[2:] = params.reshape(N - 2, 2)
                    return -_compute_min_ratio(pts)
                
                res_nm = minimize(
                    nm_objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_nm.fun < nm_objective(best_params):
                    best_params = res_nm.x.copy()
                    pts = np.empty((N, 2))
                    pts[0] = [0.0, 0.0]
                    pts[1] = [1.0, 0.0]
                    pts[2:] = best_params.reshape(N - 2, 2)
                    score = _compute_min_ratio(pts)
                    if score > best_score:
                        best_score = score
            except Exception:
                pass
    
    # Construct final points
    if best_params is None:
        best_params = _generate_initial_params(rng, 'polygon')
    
    final_pts = np.empty((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
