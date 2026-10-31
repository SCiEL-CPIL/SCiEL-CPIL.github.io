"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START
import sys

def _triangle_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns a 1D array of areas.
    """
    n = len(points)
    areas = []
    # Vectorized computation for speed
    # Get indices for all combinations
    # For n=13, 286 triangles. Looping is fine, but vectorized is better.
    # We can use broadcasting to compute all cross products at once?
    # Let's stick to a simple loop or itertools for clarity and low memory overhead for n=13.
    for i, j, k in itertools.combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        areas.append(0.5 * abs(cross))
    return np.array(areas)

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate, return 0 or small value
        return 1e-10

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-8:
        return 0.0
    return min_area / hull_area

def _objective(points_flat, tau=0.01):
    """
    Objective function for the optimizer (to be minimized).
    We want to maximize min_area / hull_area.
    We use a soft-min surrogate for the minimum area to make it smooth.
    soft_min(a) = -tau * log(sum(exp(-a / tau)))
    
    Note: We are maximizing the score, so we minimize -score.
    """
    points = points_flat.reshape((N, 2))
    
    # Ensure points are within bounds (DE handles bounds, but minimize might drift)
    # We rely on bounds in DE and L-BFGS-B.
    
    areas = _triangle_areas(points)
    hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-8:
        return 1e10  # Large penalty
    
    # Soft min of areas
    # To avoid overflow in exp, shift by max
    # exp(-a/tau) can be huge if a is small and tau is small?
    # a is positive. -a/tau is negative. exp(negative) is small.
    # If a=0, exp(0)=1.
    # If a is large, exp(-large) -> 0.
    # Sum of exps is safe.
    # However, if tau is very small, this approximates min well.
    # Let's use log-sum-exp trick for stability:
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -areas / tau
    x = -areas / tau
    max_x = np.max(x)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    soft_min_area = -tau * log_sum_exp
    
    score = soft_min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Define bounds: points in [0, 1] x [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # 1. Differential Evolution for global search
    # maxiter and popsize are tuned to fit within time.
    # N=13, dim=26.
    # DE is stochastic.
    
    best_result = None
    best_score_val = -1e10
    
    # Run DE
    try:
        # Use a reasonable population size and max iterations
        # 'tol' for convergence
        de_result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=20, 
            tol=1e-4,
            polish=False  # We will do our own polish
        )
        
        if time.time() > deadline:
            # If DE took too long, return best found so far if any, or fallback
            # DE doesn't easily return intermediate bests without callback in older scipy
            # But it returns the final best. If it timed out internally, it might return a suboptimal one.
            # We check time after.
            pass
            
        points_de = de_result.x.reshape((N, 2))
        score_de = _score(points_de)
        
        if score_de > best_score_val:
            best_score_val = score_de
            best_result = points_de
            
    except Exception as e:
        print(f"DE failed: {e}")

    if time.time() > deadline:
        if best_result is not None:
            return best_result
        # Fallback to random if DE failed immediately
        return np.random.rand(N, 2)

    # 2. Local Refinement with L-BFGS-B
    # Start from DE result
    x0 = best_result.flatten()
    
    try:
        res = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success and time.time() < deadline:
            points_res = res.x.reshape((N, 2))
            score_res = _score(points_res)
            
            if score_res > best_score_val:
                best_score_val = score_res
                best_result = points_res
    except Exception as e:
        print(f"Minimize failed: {e}")

    # 3. Optional: Nelder-Mead for final polish if time permits
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _objective, 
                best_result.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res_nm.success:
                points_nm = res_nm.x.reshape((N, 2))
                score_nm = _score(points_nm)
                
                if score_nm > best_score_val:
                    best_score_val = score_nm
                    best_result = points_nm
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")

    if best_result is None:
        # Fallback
        return np.random.rand(N, 2)
        
    return best_result

# EVOLVE-BLOCK-END
