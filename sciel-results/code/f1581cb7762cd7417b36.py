import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute indices for all 286 triples
# This is done once at module load to speed up evaluation
_triples = list(combinations(range(N), 3))
_i_idx = np.array([t[0] for t in _triples])
_j_idx = np.array([t[1] for t in _triples])
_k_idx = np.array([t[2] for t in _triples])

def _min_triangle_area_vectorized(points):
    """
    Calculate the minimum area of all triangles formed by the points.
    Vectorized for speed.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    v1x = x[_j_idx] - x[_i_idx]
    v1y = y[_j_idx] - y[_i_idx]
    v2x = x[_k_idx] - x[_i_idx]
    v2y = y[_k_idx] - y[_i_idx]
    
    # Area = 0.5 * |cross product|
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the raw score: min_triangle_area / convex_hull_area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vectorized(points)
    return min_tri / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate for the score using log-sum-exp on the negative areas.
    Minimizing this is equivalent to maximizing the soft-min of the areas.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 1e6

    x = points[:, 0]
    y = points[:, 1]
    
    v1x = x[_j_idx] - x[_i_idx]
    v1y = y[_j_idx] - y[_i_idx]
    v2x = x[_k_idx] - x[_i_idx]
    v2y = y[_k_idx] - y[_i_idx]
    
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we subtract the max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    
    # If all areas are 0, max_val is 0, sum is N, log is log(N), result is -log(N)/beta
    # If max_val is very negative, exp will underflow to 0, which is fine.
    if max_val < -700: # exp(-700) is ~1e-304, close to underflow
        return 1e6
        
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    score = soft_min_area / hull_area
    return -score

def _regular_polygon(n, center=(0.0, 0.0), radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit

    # 1. Initial Baseline: Regular 13-gon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)

    # 2. Differential Evolution (Global Search)
    # Use a short DE run to find a good candidate. 
    # The vectorized score is fast, so we can afford a small population.
    if time.time() - start_time < deadline - 60:
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        def de_objective(x):
            if time.time() - start_time > deadline:
                return 0.0
            points = x.reshape(-1, 2)
            # For DE, we can use the hard score directly as it's fast enough
            # and DE is derivative-free.
            return -_score(points)
            
        try:
            # popsize=10, maxiter=20 is a small budget but DE is stochastic.
            # We rely on the subsequent L-BFGS to refine.
            res_de = differential_evolution(
                de_objective, bounds, 
                maxiter=20, popsize=10, 
                tol=1e-4, seed=42, 
                polish=False, init='latinhypercube'
            )
            if res_de.fun < -best_score:
                candidate = res_de.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # 3. Gradient-Based Refinement (L-BFGS-B) using Soft-Min Surrogate
    # The soft-min makes the objective smooth, allowing L-BFGS to work effectively.
    # We run this from the best point found so far, and also from a few symmetric guesses.
    
    start_points = [best_points]
    
    # Add a few symmetric configurations as diverse starting points
    # 7-6 split
    pts_76 = np.vstack([_regular_polygon(7, radius=1.0), _regular_polygon(6, radius=0.5, angle_offset=np.pi/6)])
    start_points.append(pts_76)
    
    # 12-1 split
    pts_121 = np.vstack([_regular_polygon(12, radius=1.0), np.array([[0.0, 0.0]])])
    start_points.append(pts_121)
    
    # 13-0 split (Regular polygon)
    start_points.append(_regular_polygon(N))

    for start_pt in start_points:
        if time.time() - start_time > deadline - 30:
            break
            
        x0 = start_pt.flatten()
        
        # Multi-beta annealing: Start with low beta (smoother) and increase
        # to sharpen the minimum.
        current_x = x0
        for beta in [10.0, 50.0, 100.0, 200.0]:
            if time.time() - start_time > deadline - 30:
                break
                
            def soft_obj(x):
                if time.time() - start_time > deadline:
                    return 1e6
                points = x.reshape(-1, 2)
                return _soft_min_score(points, beta)
            
            try:
                res_lbfgs = minimize(
                    soft_obj, current_x, 
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_x = res_lbfgs.x
            except Exception:
                pass
        
        # Evaluate the hard score of the refined points
        candidate = current_x.reshape(-1, 2)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    # 4. Final Polishing with Nelder-Mead on Hard Score
    # The soft-min might have a slight bias. Nelder-Mead on the exact
    # non-smooth function can squeeze out the last bit of performance.
    if time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        
        def hard_obj(x):
            if time.time() - start_time > deadline:
                return 0.0
            points = x.reshape(-1, 2)
            return -_score(points)
        
        try:
            res_nm = minimize(
                hard_obj, x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                candidate = res_nm.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    return best_points
