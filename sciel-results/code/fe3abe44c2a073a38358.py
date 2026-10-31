import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]

def _get_triangle_areas(points):
    """Vectorized calculation of areas for all triples."""
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Initial configuration: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    x0 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]).flatten()
    
    best_x = x0.copy()
    best_score = _score(best_x.reshape(N, 2))
    
    # Stage 1: Global optimization via differential evolution
    if time.time() < deadline - 60:
        try:
            # Define bounds for differential evolution
            # We allow points to move in a box around the origin
            bounds = [(-2.0, 2.0)] * (N * 2)
            
            # Run differential evolution with limited evaluations to stay within time
            # The objective is the negative score
            def _de_objective(params):
                return _neg_score(params)
            
            # Use a smaller population and fewer generations to fit in time
            de_result = differential_evolution(
                _de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=42,
                polish=False,  # We will polish manually to save time
                init='sobol'
            )
            
            if de_result.x is not None:
                candidate = de_result.x.reshape(N, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = de_result.x.copy()
        except Exception as e:
            pass
    
    # Stage 2: Local refinement using Nelder-Mead on the best found so far
    if time.time() < deadline - 20:
        try:
            # Use the best point from DE as the starting point for local search
            x_start = best_x.copy()
            
            res = minimize(
                _neg_score,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-9, 'xatol': 1e-9, 'adaptive': True}
            )
            
            candidate = res.x.reshape(N, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_x = res.x.copy()
        except Exception as e:
            pass
    
    # Stage 3: Softmin-based local refinement for smoother landscape
    if time.time() < deadline - 10:
        try:
            x_start = best_x.copy()
            
            # Try different alpha values for softmin
            for alpha in [50, 20, 10, 5]:
                if time.time() > deadline - 5:
                    break
                res = minimize(
                    _neg_softmin_score,
                    x_start,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'fatol': 1e-9, 'xatol': 1e-9, 'adaptive': True}
                )
                
                candidate = res.x.reshape(N, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = res.x.copy()
                    x_start = res.x.copy()  # Update starting point for next alpha
        except Exception as e:
            pass
    
    # Stage 4: Final hard-score polish
    if time.time() < deadline - 2:
        try:
            x_start = best_x.copy()
            res = minimize(
                _neg_score,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 100, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True}
            )
            
            candidate = res.x.reshape(N, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_x = res.x.copy()
        except Exception as e:
            pass
    
    return best_x.reshape(N, 2)
