"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple indices for efficiency
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
    # Clip to avoid overflow (shifted should be <= 0)
    shifted = np.clip(shifted, -100, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
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

def _differential_evolution_step(rng, population, scores, time_limit, elapsed_start):
    """
    Perform a DE mutation step.
    population: list of param arrays (22,)
    scores: list of scores
    Returns updated population and scores.
    """
    pop_size = len(population)
    if pop_size < 4:
        return population, scores
    
    # DE/rand/1 mutation
    new_member_idx = rng.integers(0, pop_size)
    a = rng.choice([i for i in range(pop_size) if i != new_member_idx])
    b = rng.choice([i for i in range(pop_size) if i != new_member_idx and i != a])
    c = rng.choice([i for i in range(pop_size) if i != new_member_idx and i != a and i != b])
    
    F = rng.uniform(0.5, 1.0)
    trial = population[a] + F * (population[b] - population[c])
    
    # Evaluate trial
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 1.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = trial.reshape(N - 2, 2)
    score = _compute_min_ratio(pts)
    
    if score > scores[new_member_idx]:
        population[new_member_idx] = trial
        scores[new_member_idx] = score
    
    return population, scores

def _nelder_mead_polish(params, time_budget=5.0, maxiter=2000):
    """
    Nelder-Mead polish on the exact objective (negative min ratio).
    """
    def _exact_obj(p):
        pts = np.empty((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = p.reshape(N - 2, 2)
        return -_compute_min_ratio(pts)
    
    res = minimize(
        _exact_obj,
        params,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
    )
    return res.x

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    num_starts = 12
    starts = []
    start_methods = ['polygon', 'halton', 'random', 'polygon', 'halton', 'random',
                     'polygon', 'halton', 'random', 'polygon', 'halton', 'random']
    
    for i in range(num_starts):
        if (time.time() - start_time) > time_limit - 20.0:
            break
        method = start_methods[i % len(start_methods)]
        p = _generate_initial_params(rng, method)
        starts.append(p)
    
    # Phase 2: Local optimization with L-BFGS-B using smooth min
    # Use multiple tau values for annealing
    tau_schedule = [20, 50, 150, 400, 1000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 5.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 3.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Evaluate true score of the result
                new_params = res.x
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = new_params.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = new_params.copy()
                
                # Update the start for next tau iteration
                starts[i] = new_params.copy()
                
            except Exception:
                pass
    
    # Phase 3: Final polish with high tau on the best solution found
    if best_params is not None and (time.time() - start_time) < time_limit - 10.0:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(5000,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = res.x.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: DE-inspired global search with local polish
    if (time.time() - start_time) < time_limit - 15.0:
        # Create population from best starts
        population = [starts[i].copy() for i in range(min(6, len(starts)))]
        scores = []
        for p in population:
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = p.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_params = p.copy()
        
        # Run DE for a few iterations
        max_de_iters = 20
        for it in range(max_de_iters):
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            population, scores = _differential_evolution_step(rng, population, scores, time_limit, start_time)
            
            # Track best
            for i, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_params = population[i].copy()
    
    # Phase 5: Nelder-Mead polish on the best solution and top candidates
    candidates_to_polish = []
    if best_params is not None:
        candidates_to_polish.append(best_params)
    
    # Add top candidates from 'starts' (which were updated by L-BFGS-B)
    # Sort starts by their last known score (approximated by re-evaluation or stored)
    # For efficiency, we just take the first few unique ones from starts
    for i in range(min(5, len(starts))):
        # Check if this start is significantly different from best_params
        if np.linalg.norm(starts[i] - best_params) > 0.01:
            candidates_to_polish.append(starts[i])
            
    # Limit to 5 candidates to save time
    candidates_to_polish = candidates_to_polish[:5]
    
    for cand in candidates_to_polish:
        if (time.time() - start_time) > time_limit - 5.0:
            break
        try:
            polished = _nelder_mead_polish(cand, maxiter=3000)
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = polished.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_params = polished.copy()
        except Exception:
            pass

    # Phase 6: Multi-start L-BFGS-B polish with perturbations (Basin Hopping)
    if (time.time() - start_time) < time_limit - 2.0:
        num_perturb = 10
        for i in range(num_perturb):
            if (time.time() - start_time) > time_limit - 2.0:
                break
            try:
                # Smaller perturbations for fine-tuning, larger for escaping
                sigma = 0.005 if i < 5 else 0.02
                perturbed = best_params + rng.normal(0, sigma, 22)
                # Use high tau for precise local optimization
                res = minimize(
                    _objective,
                    perturbed,
                    args=(2000,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = res.x.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Reconstruct final points
    if best_params is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    final_pts = np.empty((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
