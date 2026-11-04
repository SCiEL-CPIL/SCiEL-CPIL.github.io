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

def _normalize_affine(pts):
    """
    Apply an affine transformation to pts such that:
    pts[0] -> (0,0)
    pts[1] -> (1,0)
    pts[2] -> (0,1)
    Returns the transformed points or None if the first 3 points are collinear.
    """
    # Check for collinearity of first 3 points
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(det) < 1e-9:
        return None
    
    # Affine transform: P' = A P + t
    # We want:
    # A p0 + t = (0,0)  => t = -A p0
    # A p1 - A p0 = (1,0)
    # A p2 - A p0 = (0,1)
    
    # Let M = [p1-p0, p2-p0] (2x2 matrix with these as columns)
    M = np.column_stack((v1, v2))
    # We want A * M = I (identity)
    # So A = inv(M)
    A = np.linalg.inv(M)
    
    # t = -A * p0
    t = -A @ pts[0]
    
    # Apply to all points
    pts_new = (A @ pts.T).T + t
    
    return pts_new

def _params_to_points(params):
    """
    Convert 20 parameters (10 points * 2 coords) to 13 points.
    P0, P1, P2 are fixed at (0,0), (1,0), (0,1).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.0, 1.0]
    pts[3:] = params.reshape(N - 3, 2)
    return pts

def _points_to_params(pts):
    """
    Convert 13 points to 20 parameters.
    Assumes pts are already normalized (P0=(0,0), P1=(1,0), P2=(0,1)).
    """
    return pts[3:].flatten()

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(ratios).
    params: 20 values representing p_4 ... p_13.
    p_1, p_2, p_3 are fixed.
    """
    pts = _params_to_points(params)
    
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

def _generate_normalized_starts(rng, n_starts=20):
    """
    Generate diverse starting points, normalize them affinely, and return params.
    """
    starts = []
    for _ in range(n_starts):
        # Generate random points in a square
        pts = rng.uniform(-1, 1, (N, 2))
        
        # Ensure they are in convex position? 
        # Random points might not be convex. 
        # For initialization, we can just use them, but the optimizer might struggle if they are not convex.
        # Better: Generate points on a circle with noise, then normalize.
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.1, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Shuffle to ensure P0, P1, P2 are likely non-collinear and well-spaced
        idx = rng.permutation(N)
        pts = pts[idx]
        
        norm_pts = _normalize_affine(pts)
        if norm_pts is not None:
            # Check if the normalized points are reasonable (not extremely skewed)
            # The convex hull should contain the origin or be reasonable
            params = _points_to_params(norm_pts)
            starts.append(params)
            
    return starts

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points using affine normalization
    starts = _generate_normalized_starts(rng, n_starts=25)
    
    # Also add a regular polygon start
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Shuffle to pick 3 non-collinear
    idx = rng.permutation(N)
    reg_pts = reg_pts[idx]
    norm_reg = _normalize_affine(reg_pts)
    if norm_reg is not None:
        starts.append(_points_to_params(norm_reg))
        
    # Evaluate initial scores and sort
    if not starts:
        # Fallback
        starts = [rng.uniform(-1, 1, 20)]
        
    start_scores = []
    for p in starts:
        pts = _params_to_points(p)
        score = _compute_min_ratio(pts)
        start_scores.append(score)
        
    sorted_indices = np.argsort(start_scores)[::-1]
    # Keep top 15 starts for optimization
    num_starts = min(15, len(starts))
    active_starts = [starts[idx] for idx in sorted_indices[:num_starts]]
    
    # Phase 2: Multi-scale tau annealing with L-BFGS-B
    # Since the search space is reduced (20 vs 22 params) and affine invariance is removed,
    # convergence should be faster.
    tau_schedule = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        
        for i in range(len(active_starts)):
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            try:
                # Adaptive maxiter
                remaining = time_limit - (time.time() - start_time)
                maxiter = min(300, max(50, int(remaining * 3)))
                
                res = minimize(
                    _objective,
                    active_starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Evaluate true score
                new_params = res.x
                pts = _params_to_points(new_params)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = new_params.copy()
                
                # Update start for next iteration
                active_starts[i] = new_params.copy()
                
            except Exception:
                pass
    
    # Phase 3: Basin Hopping / Random Restarts
    # Use the best found so far to generate new random perturbations in the normalized space
    if (time.time() - start_time) < time_limit - 20.0:
        # Create a population from the best few starts
        current_scores = []
        for p in active_starts:
            pts = _params_to_points(p)
            current_scores.append(_compute_min_ratio(pts))
            
        sorted_indices = np.argsort(current_scores)[::-1]
        pop_size = min(10, len(active_starts))
        population = [active_starts[idx].copy() for idx in sorted_indices[:pop_size]]
        scores = [current_scores[idx] for idx in sorted_indices[:pop_size]]
        
        max_de_iters = 50
        for it in range(max_de_iters):
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            # DE mutation
            new_idx = rng.integers(0, pop_size)
            candidates = [i for i in range(pop_size) if i != new_idx]
            if len(candidates) < 3:
                break
            a = rng.choice(candidates)
            b = rng.choice([i for i in candidates if i != a])
            c = rng.choice([i for i in candidates if i != a and i != b])
            
            F = rng.uniform(0.5, 1.0)
            trial = population[a] + F * (population[b] - population[c])
            
            # Evaluate
            pts = _params_to_points(trial)
            score = _compute_min_ratio(pts)
            
            if score > scores[new_idx]:
                population[new_idx] = trial.copy()
                scores[new_idx] = score
                
                # Local polish
                if (time.time() - start_time) < time_limit - 15.0:
                    try:
                        res = minimize(
                            _objective,
                            trial,
                            args=(1000,),
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12}
                        )
                        pts2 = _params_to_points(res.x)
                        score2 = _compute_min_ratio(pts2)
                        if score2 > scores[new_idx]:
                            population[new_idx] = res.x.copy()
                            scores[new_idx] = score2
                            if score2 > best_score:
                                best_score = score2
                                best_params = res.x.copy()
                    except Exception:
                        pass

    # Phase 4: Final high-precision polish
    if best_params is not None and (time.time() - start_time) < time_limit - 5.0:
        for tau in [2000, 5000, 10000]:
            if (time.time() - start_time) > time_limit - 3.0:
                break
            try:
                res = minimize(
                    _objective,
                    best_params,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts = _params_to_points(res.x)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
        
        # Nelder-Mead final touch
        if (time.time() - start_time) < time_limit - 2.0:
            try:
                def nm_obj(p):
                    pts = _params_to_points(p)
                    return -_compute_min_ratio(pts)
                
                res_nm = minimize(
                    nm_obj,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts = _params_to_points(res_nm.x)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_params = res_nm.x.copy()
            except Exception:
                pass

    if best_params is None:
        best_params = starts[0]
        
    final_pts = _params_to_points(best_params)
    
    return final_pts

# EVOLVE-BLOCK-END
