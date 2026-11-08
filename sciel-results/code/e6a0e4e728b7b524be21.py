"""
Strategy:
1. Symmetry-Fixed Representation:
   Represent points as deviations from a regular N-gon.
   params = [radial_devs (N), angular_devs (N)]
   This reduces the effective search space by fixing the center and average scale,
   removing the translational and rotational degrees of freedom that cause
   flat directions in the Cartesian landscape.
   
2. Global Search via Differential Evolution (DE):
   DE is robust to the rugged landscape of the min-triangle-area function.
   We maximize the "hard" score (min_area / hull_area) directly, but use a
   slightly smoothed objective if needed. However, for N=14, the hard objective
   is usually sufficient for DE if the population is diverse enough.
   
3. Local Refinement via L-BFGS-B:
   After DE converges or time expires, we run L-BFGS-B on the best candidate
   found by DE. L-BFGS-B is very effective for the final "polishing" phase where
   the landscape is smooth enough (locally convex-like near the optimum).

4. Time Management:
   Strict deadline guards. 70% of time for DE, 30% for L-BFGS.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14
DEADLINE = 145.0  # Safety margin from 150s limit

# Precompute triangle indices for speed
def _generate_triangle_indices(n):
    inds = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                inds.append((i, j, k))
    return np.array(inds, dtype=int)

_TRI_INDICES = _generate_triangle_indices(N)
_I, _J, _K = _TRI_INDICES.T

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(pts):
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(pts):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 0.0
        
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
        
    return min_area / hull_a

def _params_to_points(params):
    """
    Convert parameters to points.
    params[0:N] are radial deviations (added to base radius 1.0)
    params[N:2N] are angular deviations (added to base angles)
    
    Base configuration: Regular polygon of radius 1.
    """
    base_angles = 2 * np.pi * np.arange(N) / N
    radii = 1.0 + params[:N]
    angles = base_angles + params[N:]
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _neg_score_from_params(params):
    """
    Objective function for optimization: minimize negative score.
    """
    pts = _params_to_points(params)
    return -_score(pts)

def _local_refine_params(params, time_limit):
    """
    Refine parameters using L-BFGS-B.
    """
    start_t = time.time()
    
    def objective(p):
        if time.time() - start_t > time_limit:
            raise TimeoutError("Local refinement timeout")
        return _neg_score_from_params(p)
    
    try:
        res = minimize(objective, params, method='L-BFGS-B', 
                       options={'maxiter': 100, 'disp': False, 'maxls': 30})
        # Check if the result is better than the input
        if _score(_params_to_points(res.x)) > _score(_params_to_points(params)):
            return res.x
    except Exception:
        pass
    return params

def construct():
    start_time = time.time()
    
    # Define bounds for DE
    # Radial deviations: points should stay roughly within a reasonable range
    # Angular deviations: shouldn't cross each other too much, but allow flexibility
    # Base radius is 1.0.
    # If a point moves too far in/out, hull area changes drastically.
    # Let's allow radial in [-0.5, 0.5] and angular in [-pi/14, pi/14] initially?
    # Actually, for N=14, the angle between neighbors is 2*pi/14 ~ 0.44 rad.
    # A deviation of +/- 0.2 rad is about half the gap.
    
    bounds = [(-0.5, 0.5)] * N + [(-0.3, 0.3)] * N
    
    # Initial guess: Regular polygon (all zeros)
    initial_params = np.zeros(2 * N)
    
    # We will run DE. 
    # To ensure we don't timeout, we set a callback to check time.
    def de_callback(xk, convergence):
        if time.time() > DEADLINE - 30:
            raise TimeoutError("DE timeout")
        return False

    best_params = initial_params
    best_score = _score(_params_to_points(best_params))
    
    # Run Differential Evolution
    # popsize=15 is a good default for 28 variables.
    # maxiter is set high, but we rely on the callback for time.
    try:
        de_result = differential_evolution(
            _neg_score_from_params, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=20, 
            tol=1e-5, 
            callback=de_callback,
            polish=False  # We will do our own polishing with L-BFGS which is more controlled
        )
        
        if _score(_params_to_points(de_result.x)) > best_score:
            best_score = _score(_params_to_points(de_result.x))
            best_params = de_result.x.copy()
            
    except TimeoutError:
        pass
    except Exception as e:
        # If DE fails completely, we might still have best_params from initialization
        pass

    # Local Refinement Phase
    # We have ~30 seconds left.
    # We can try to refine the best solution found by DE.
    # We can also try a few random restarts of L-BFGS from different initial points
    # if we have time, but usually polishing the DE result is the best use of time.
    
    # Strategy: 
    # 1. Polish the best DE result.
    # 2. If time remains, try polishing a slightly perturbed version to escape local minima.
    
    remaining_time = DEADLINE - time.time()
    
    if remaining_time > 5:
        # Polish best_params
        refined_params = _local_refine_params(best_params, time_limit=remaining_time - 5)
        if _score(_params_to_points(refined_params)) > best_score:
            best_score = _score(_params_to_points(refined_params))
            best_params = refined_params.copy()
            
    # If we still have significant time, try one more restart with a different seed
    # or a perturbed version of the best solution.
    remaining_time = DEADLINE - time.time()
    if remaining_time > 10:
        # Perturb the best solution slightly and polish
        perturbed = best_params + np.random.normal(0, 0.05, size=2*N)
        # Clip to bounds roughly
        for i in range(N):
            perturbed[i] = np.clip(perturbed[i], -0.5, 0.5)
            perturbed[N+i] = np.clip(perturbed[N+i], -0.3, 0.3)
            
        refined_params2 = _local_refine_params(perturbed, time_limit=remaining_time - 5)
        if _score(_params_to_points(refined_params2)) > best_score:
            best_score = _score(_params_to_points(refined_params2))
            best_params = refined_params2.copy()

    # Final check
    final_pts = _params_to_points(best_params)
    
    # One last tiny polish if we have > 2 seconds
    remaining_time = DEADLINE - time.time()
    if remaining_time > 2:
        # Just a quick L-BFGS step on the final points directly? 
        # No, stick to params for consistency.
        pass
        
    return final_pts

# EVOLVE-BLOCK-END
