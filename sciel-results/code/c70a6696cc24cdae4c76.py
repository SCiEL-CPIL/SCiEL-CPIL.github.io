"""Heilbronn convex regions n=14: simulated annealing over angles on unit circle."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Vectorized triangle areas for all C(n,3) triples."""
    n = points.shape[0]
    # Precompute all index triples
    idx = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area_from_angles(angles):
    """Min triangle area given angles on unit circle."""
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.min(_triangle_areas(points))

def _score_from_angles(angles):
    """Score = min triangle area / convex hull area."""
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        return _min_triangle_area_from_angles(angles) / hull_area
    except Exception:
        return 0.0

def _negative_min_area(angles):
    """Objective for minimization: negative of min triangle area."""
    return -_min_triangle_area_from_angles(angles)

def _powell_polish(angles, maxiter=5000, tol=1e-10):
    """Polish angles with Powell. Returns (score, angles)."""
    try:
        res = minimize(_negative_min_area, angles, method='Powell',
                       options={'maxiter': maxiter, 'ftol': tol, 'xtol': tol})
        score = -res.fun
        if score > 0 and not np.any(np.isnan(res.x)):
            return score, res.x
    except Exception:
        pass
    return _min_triangle_area_from_angles(angles), angles

def _simulated_annealing(start_angles, n_steps, T0, T1, rng, deadline):
    """
    Simulated annealing over angles.
    start_angles: initial (N,) array
    n_steps: number of SA steps
    T0, T1: initial and final temperature
    rng: numpy Generator
    deadline: absolute time to stop
    Returns best (score, angles) found.
    """
    best_score = _min_triangle_area_from_angles(start_angles)
    best_angles = start_angles.copy()
    current_angles = start_angles.copy()
    current_score = best_score
    
    # Geometric cooling
    if n_steps > 1:
        cool_factor = (T1 / T0) ** (1.0 / n_steps)
    else:
        cool_factor = 1.0
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        T = T0 * (cool_factor ** step)
        
        # Perturb all angles with Gaussian noise
        noise_scale = 0.1 * T
        new_angles = current_angles + rng.normal(0, noise_scale, N)
        
        new_score = _min_triangle_area_from_angles(new_angles)
        
        # Accept if better, or with probability exp((new - current)/T)
        if new_score >= current_score:
            accept = True
        else:
            delta = new_score - current_score  # negative
            # delta/T is negative, so exp is < 1
            if T > 1e-12:
                accept = rng.random() < np.exp(delta / T)
            else:
                accept = False
        
        if accept:
            current_angles = new_angles
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_angles = new_angles.copy()
    
    return best_score, best_angles

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (limit 150s)
    
    rng = np.random.default_rng(2024)
    
    best_score = -np.inf
    best_angles = None
    
    # Regular polygon as initial guess
    regular_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_score = _min_triangle_area_from_angles(regular_angles)
    best_score = reg_score
    best_angles = regular_angles.copy()
    
    # Multiple SA runs from diverse starts
    # Each SA run: ~10000 steps, T from 0.5 to 0.001
    n_sa_runs = 8
    n_steps_per_run = 12000
    T0 = 0.5
    T1 = 0.001
    
    # Diverse starting points
    starts = []
    starts.append(regular_angles.copy())
    
    # Jittered regular polygons
    for jitter in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
        starts.append(regular_angles + rng.normal(0, jitter, N))
    
    # Random
    for _ in range(4):
        starts.append(rng.uniform(0, 2*np.pi, N))
    
    # Run SA from each start
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:  # Leave time for polishing
            break
        
        # Scale steps based on remaining time
        remaining = deadline - 30 - time.time()
        runs_left = len(starts) - i
        time_per_run = remaining / max(runs_left, 1)
        # Estimate ~0.0005s per SA step (vectorized)
        steps = int(min(n_steps_per_run, max(2000, time_per_run / 0.0005)))
        
        score, angles = _simulated_annealing(start, steps, T0, T1, rng, deadline - 30)
        
        if score > best_score:
            best_score = score
            best_angles = angles.copy()
    
    # Polish the best result with Powell
    if time.time() < deadline - 5:
        polish_score, polish_angles = _powell_polish(best_angles, maxiter=20000, tol=1e-12)
        if polish_score > best_score:
            best_score = polish_score
            best_angles = polish_angles
    
    # Final score computation
    final_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    # Verify score
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        min_area = _min_triangle_area_from_angles(best_angles)
        final_score = min_area / hull_area
    except Exception:
        final_score = 0.0
    
    return final_points
# EVOLVE-BLOCK-END
