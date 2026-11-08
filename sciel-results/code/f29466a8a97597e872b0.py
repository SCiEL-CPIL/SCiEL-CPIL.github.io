"""
Strategy:
1. Represent points as (r, theta) where theta is perturbed from a regular polygon.
   This reduces the search space and provides a better inductive bias.
2. Use Simulated Annealing for global exploration to avoid local minima.
3. Use L-BFGS-B for local refinement on the raw coordinates to fine-tune the exact positions.
4. The objective is to maximize min_triangle_area / hull_area.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
DEADLINE = 145.0

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

def _neg_score(pts):
    """
    Negative score for minimization.
    """
    return -_score(pts)

def _params_to_points(params):
    """
    Convert parameters to points.
    params[0:N] are radial perturbations (multipliers on base radius 1.0)
    params[N:2N] are angular perturbations (radians added to base angles)
    
    Base configuration: Regular polygon of radius 1.
    """
    base_angles = 2 * np.pi * np.arange(N) / N
    radii = 1.0 + params[:N]
    angles = base_angles + params[N:]
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _sa_step(pts, temp, time_left):
    """
    Perform a single step of simulated annealing.
    """
    if time.time() > time_left:
        return pts, False
        
    current_score = _score(pts)
    
    # Generate a neighbor
    # Perturb random points slightly
    n_perturb = np.random.randint(1, 4) # 1 to 3 points
    idx = np.random.choice(N, n_perturb, replace=False)
    
    new_pts = pts.copy()
    # Random walk in Cartesian space
    step_size = 0.05 * temp # Scale step with temperature
    new_pts[idx] += np.random.normal(0, step_size, size=(n_perturb, 2))
    
    new_score = _score(new_pts)
    
    # Acceptance probability
    if new_score > current_score:
        return new_pts, True
    else:
        # delta is negative
        delta = new_score - current_score
        if temp > 1e-8:
            prob = np.exp(delta / temp)
            if np.random.random() < prob:
                return new_pts, True
    return pts, False

def _local_refine(pts, time_limit):
    """
    Refine points using L-BFGS-B.
    We minimize -score(pts).
    """
    start_t = time.time()
    
    def objective(x):
        if time.time() - start_t > time_limit:
            raise TimeoutError("Local refinement timeout")
        p = x.reshape((N, 2))
        return _neg_score(p)
    
    try:
        res = minimize(objective, pts.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 50, 'disp': False, 'maxls': 20})
        if res.success or _score(res.x.reshape((N,2))) > _score(pts):
            return res.x.reshape((N, 2))
    except Exception:
        pass
    return pts

def construct():
    start_time = time.time()
    
    # Initial guess: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_pts)
    
    # Parameters for SA
    # We will work directly on Cartesian coordinates for the SA to allow 
    # more flexible deformations than just radial/angular, but we can 
    # re-center and normalize occasionally to prevent drift.
    
    # Actually, let's stick to Cartesian for SA but use a smart initialization.
    # The regular polygon is a very strong local minimum.
    # We need to break symmetry.
    
    # Run multiple SA restarts with different initial perturbations
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize with regular polygon + small random noise
        noise_scale = 0.1 if r == 0 else 0.3
        pts = best_pts + np.random.normal(0, noise_scale, size=(N, 2))
        
        # Center and normalize scale for stability
        pts = pts - np.mean(pts, axis=0)
        hull_a = _hull_area(pts)
        if hull_a > 0:
            pts = pts * (1.0 / np.sqrt(hull_a))
            
        current_score = _score(pts)
        
        # Simulated Annealing
        temp = 0.5
        cooling_rate = 0.995
        sa_start = time.time()
        
        while time.time() < DEADLINE - 15 and temp > 0.01:
            pts, _ = _sa_step(pts, temp, DEADLINE - 15)
            temp *= cooling_rate
            
            # Periodic normalization to prevent drift
            if r > 0 and np.random.random() < 0.05:
                pts = pts - np.mean(pts, axis=0)
                hull_a = _hull_area(pts)
                if hull_a > 1e-6:
                    pts = pts * (1.0 / np.sqrt(hull_a))
            
            if _score(pts) > current_score:
                current_score = _score(pts)
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    
        # Local refinement on the best point found in this SA run
        if time.time() < DEADLINE - 5:
            refined_pts = _local_refine(pts, time_limit=min(5.0, DEADLINE - 5 - time.time()))
            refined_score = _score(refined_pts)
            if refined_score > current_score:
                current_score = refined_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = refined_pts.copy()
                    
    # Final global polish on the absolute best
    if time.time() < DEADLINE - 2:
        final_pts = _local_refine(best_pts, time_limit=min(3.0, DEADLINE - 2 - time.time()))
        if _score(final_pts) > best_score:
            best_pts = final_pts
            
    return best_pts

# EVOLVE-BLOCK-END
