import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations
from scipy.optimize import differential_evolution

N = 13

# Pre-compute triangle indices for N=13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)

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
        return 0.0

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(x):
    """
    Objective function for minimization.
    Returns negative score (higher score is better, so we minimize -score).
    Returns a large penalty if the configuration is invalid (collinear, etc.)
    """
    points = x.reshape((N, 2))
    score = calculate_score(points)
    if score <= 0:
        return 1e10
    return -score

def _local_polish(points, time_limit=10.0):
    """
    Perform a local coordinate-wise search to refine the best point found by DE.
    """
    start_time = time.time()
    current_pts = points.copy()
    current_score = calculate_score(current_pts)
    
    step = 0.01
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < start_time + time_limit and no_improve < max_no_improve:
        improved = False
        # Try moving each point in 4 directions
        for i in range(N):
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                trial_pts = current_pts.copy()
                trial_pts[i, 0] += dx
                trial_pts[i, 1] += dy
                trial_score = calculate_score(trial_pts)
                
                if trial_score > current_score:
                    current_pts = trial_pts
                    current_score = trial_score
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            step *= 0.95
            if step < 1e-6:
                break
        
        # Occasionally try larger random jumps to escape local minima
        if no_improve % 10 == 0:
            np.random.seed(int(time.time() * 1000) % 100000)
            jump = np.random.normal(0, step * 2, (N, 2))
            trial_pts = current_pts + jump
            trial_score = calculate_score(trial_pts)
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
                no_improve = 0
                
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Define bounds for DE: points within a square [-1, 1] x [-1, 1]
    # This covers a large enough area for the convex hull to be non-degenerate
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    # Run Differential Evolution
    # maxiter is set high, but we rely on the timeout mechanism if possible, 
    # or we can wrap it to check time. SciPy DE doesn't natively support time-based stopping easily,
    # so we set a reasonable maxiter and popsize.
    # With 26 dimensions, popsize=30 means 30*26=780 evaluations per generation.
    # 100 generations -> 78,000 evaluations.
    # Each evaluation is ~0.1ms (very fast vectorized).
    # 78,000 * 0.1ms = 7.8 seconds. This is safe.
    
    try:
        # Use a fixed seed for reproducibility in DE? The task allows random, but reproducibility helps debugging.
        # However, for exploration, random seed might find better global minima.
        # Let's use a fixed seed for the first run to ensure it works, then maybe randomize if time permits.
        # Actually, the task says "random" is fine. Let's use a seed that works well.
        seed = 42
        
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=30, 
            tol=1e-6, 
            seed=seed,
            polish=False # We do our own polish
        )
        
        if result.success and result.fun < 1e9:
            candidate_pts = result.x.reshape((N, 2))
            candidate_score = calculate_score(candidate_pts)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_pts.copy()
                
    except Exception as e:
        # If DE fails, fallback to a structured initialization
        pass
    
    # If DE didn't find anything good, or if we have time, try a few other seeds for DE
    # This helps with robustness.
    remaining_time = deadline - time.time()
    if remaining_time > 20.0 and (best_score < 30.0 or best_points is None):
        for seed in [123, 456, 789]:
            if time.time() > deadline - 15.0:
                break
            try:
                result = differential_evolution(
                    _objective, 
                    bounds, 
                    maxiter=20, 
                    popsize=20, 
                    tol=1e-5, 
                    seed=seed,
                    polish=False
                )
                
                if result.success and result.fun < 1e9:
                    candidate_pts = result.x.reshape((N, 2))
                    candidate_score = calculate_score(candidate_pts)
                    
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_pts.copy()
                        
            except Exception:
                pass

    # Fallback: If still no good points, use regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)
        
    # Final Polish
    if time.time() < deadline - 5.0:
        polished_pts, polished_score = _local_polish(best_points, time_limit=10.0)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts

    return best_points
