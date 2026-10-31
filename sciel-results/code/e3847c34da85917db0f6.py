import numpy as np
import time
import scipy.spatial as spat
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 13
# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min_objective(params, temperature=0.05):
    """
    Objective function for optimization.
    We want to MAXIMIZE min_area / hull_area.
    So we MINIMIZE -min_area / hull_area.
    
    To make it smooth, we approximate min(areas) with -T * log(sum(exp(-area/T))).
    This is a lower bound on min(areas) that becomes tight as T -> 0.
    
    Note: DE minimizes the function. 
    """
    points = params.reshape((N, 2))
    
    # Check for degenerate hull (collinear or very small area)
    # We can skip explicit hull check in the fast objective if we rely on the penalty,
    # but a quick check is safer. However, ConvexHull is slow.
    # Let's just compute areas. If hull is degenerate, areas will be small, score low.
    # But division by zero is bad.
    
    areas = _triangle_areas(points)
    
    # If any area is 0, the smooth min will be 0 (or close).
    # We need to handle the hull area. 
    # Using ConvexHull in the objective is expensive. 
    # Alternative: Assume points are normalized to unit area during search?
    # No, the score is min_area / hull_area.
    # If we fix hull_area to 1 by normalizing the points in the objective, 
    # then maximizing min_area is equivalent.
    
    # Normalization strategy:
    # 1. Compute hull area.
    # 2. Scale points so hull area is 1.
    # 3. Compute min area.
    # This keeps the search space well-behaved.
    
    try:
        hull = spat.ConvexHull(points)
        h_area = hull.volume
        if h_area < 1e-6:
            return 1e6 # Penalty
        # Scale points to unit hull area
        scale_factor = np.sqrt(1.0 / h_area)
        scaled_points = points * scale_factor
    except Exception:
        return 1e6 # Penalty for degenerate points
        
    areas_scaled = _triangle_areas(scaled_points)
    
    # Smooth min approximation
    # soft_min = -T * log(sum(exp(-area/T)))
    # To avoid overflow/underflow in exp, we can use logsumexp from scipy.special
    # But standard math is fine if T is not too small.
    
    # Actually, for DE, a simple penalty-based approach on the exact min might be robust enough
    # if the population is diverse. But smooth is better for gradients if we used gradient methods.
    # DE is stochastic, so smoothness is less critical but helps convergence.
    
    # Let's use a simpler smooth min:
    # min_approx = -temperature * np.log(np.sum(np.exp(-areas_scaled / temperature)))
    
    # If temperature is small, this is close to min.
    # We want to maximize this, so minimize -min_approx.
    
    # Check for numerical stability
    max_exp_arg = -np.min(areas_scaled) / temperature
    if max_exp_arg > 700: # exp overflow
         return 1e6
        
    exp_terms = np.exp(-areas_scaled / temperature)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 1e6
        
    soft_min = -temperature * np.log(sum_exp)
    
    return -soft_min

def _exact_objective(params):
    """
    Exact objective for final polishing.
    Minimizes -calculate_score(points).
    """
    points = params.reshape((N, 2))
    score = calculate_score(points)
    return -score

def _local_polish(points, time_budget):
    """
    Perform local polishing using Nelder-Mead on the exact objective.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_params = points.flatten()
    current_score = calculate_score(points)
    
    # Run Nelder-Mead
    try:
        res = minimize(
            _exact_objective, 
            current_params, 
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 500}
        )
        if time.time() < deadline:
             polished_points = res.x.reshape((N, 2))
             polished_score = calculate_score(polished_points)
             if polished_score > current_score:
                 return polished_points, polished_score
    except Exception:
        pass
        
    return points, current_score

def _generate_initial_configs():
    """
    Generate diverse initial configurations for DE population seeding or restarts.
    DE handles its own population, but we can use multiple restarts of DE
    or seed it with good guesses.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Two-ring: 1 center, 12 outer
    center = np.array([[0, 0]])
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer12 = np.column_stack((np.cos(angles12), np.sin(angles12)))
    configs.append(np.vstack([center, outer12]))
    
    # 3. Two-ring: 2 inner, 11 outer
    angles2 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    inner2 = 0.5 * np.column_stack((np.cos(angles2), np.sin(angles2)))
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    outer11 = np.column_stack((np.cos(angles11), np.sin(angles11)))
    configs.append(np.vstack([inner2, outer11]))
    
    # 4. Random points in disk (normalized)
    for _ in range(2):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        # Normalize to unit area roughly
        try:
            hull = spat.ConvexHull(pts)
            pts = pts * np.sqrt(1.0 / hull.volume)
        except:
            pass
        configs.append(pts)

    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Differential Evolution Parameters
    # Bounds: points should be within [-2, 2] to allow for scaling/normalization
    # Since we normalize in the objective, the absolute scale doesn't matter for DE,
    # but it helps to keep numbers manageable.
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # We will run DE with a limited number of iterations and then polish.
    # DE is expensive, so we limit population size and generations.
    
    # Strategy: 
    # 1. Run DE for a fixed time budget (e.g., 100s) to find a good basin.
    # 2. Use the best result from DE as a seed for local polishing.
    # 3. If time remains, try another DE run with different parameters or seeds.
    
    # To ensure we don't timeout, we use a callback to monitor time.
    # However, scipy's DE doesn't have a built-in time limit, only maxiter/maxfev.
    # We can estimate maxfev based on time.
    # One evaluation of _smooth_min_objective takes ~1-2ms (including ConvexHull).
    # Let's say 1ms per eval. 100s -> 100,000 evals.
    # DE with pop=50, gen=100 -> 5000 evals. That's fast.
    # We can afford more.
    
    # Let's run DE for a while, then polish.
    
    # We'll do 2 rounds of DE with different temperatures/strategies if time permits.
    
    configs = _generate_initial_configs()
    
    # Use the best initial config as a starting hint? DE is global, so it doesn't strictly need it,
    # but initializing the population with some good points can help.
    # scipy.optimize.differential_evolution accepts 'init' as 'latinhypercube' or a matrix.
    # Let's use a mix of random and our good configs.
    
    # Construct initial population
    pop_size = 50
    init_pop = np.random.uniform(-2, 2, size=(pop_size, 2*N))
    # Inject our good configs into the population
    for i, cfg in enumerate(configs):
        if i < pop_size:
            init_pop[i] = cfg.flatten()
            
    # Define a wrapper to check time
    # Since DE runs in a tight loop, it's hard to interrupt it cleanly in Python without signals.
    # Instead, we estimate the number of function evaluations we can afford.
    
    # Measure time for one objective call
    t0 = time.time()
    _smooth_min_objective(init_pop[0].reshape(N, 2), 0.05)
    t_eval = time.time() - t0
    if t_eval < 1e-5: t_eval = 1e-5 # Floor
    
    remaining_time = deadline - time.time() - 10.0 # Leave 10s for polishing
    if remaining_time < 5.0:
        # Not enough time for DE, just use best initial config
        for cfg in configs:
            score = calculate_score(cfg)
            if score > best_score:
                best_score = score
                best_points = cfg
        return best_points if best_points is not None else configs[0]
        
    max_evals = int(remaining_time / t_eval)
    
    # DE parameters
    # popsize is multiplier of dimension. 50 is reasonable.
    # maxiter: we will let maxfev control it.
    
    try:
        res = differential_evolution(
            _smooth_min_objective, 
            bounds, 
            args=(0.05,), # Temperature
            seed=42,
            popsize=50,
            maxiter=100,
            maxfev=max_evals,
            tol=1e-8,
            init=init_pop,
            polish=False # We will do our own polishing
        )
        
        if res.success or res.x is not None:
            de_points = res.x.reshape((N, 2))
            de_score = calculate_score(de_points)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
                
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to best initial config
        for cfg in configs:
            score = calculate_score(cfg)
            if score > best_score:
                best_score = score
                best_points = cfg

    # Polishing phase
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 5.0:
            try:
                polished_pts, polished_score = _local_polish(best_points, remaining - 2.0)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_pts
            except Exception:
                pass

    # If we still have time and score is low, try a second DE run with different temp?
    # Or just stick with the best found.
    
    if best_points is None:
        # Absolute fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
