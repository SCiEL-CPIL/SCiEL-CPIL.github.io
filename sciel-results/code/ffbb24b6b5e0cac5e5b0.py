"""Improved packing of 50 equal spheres in a unit cube using Repulsive Dynamics + L-BFGS-B."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score using KDTree for speed."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_score(centers, beta=50.0):
    """Differentiable soft-minimum score for gradient-based optimization."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    a = min_pair / 2.0
    b = min_wall
    # Soft min: -1/beta * log(exp(-beta*a) + exp(-beta*b))
    m = min(a, b)
    return m - math.log(math.exp(-beta * (a - m)) + math.exp(-beta * (b - m))) / beta

def _objective_flat(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft-min score."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    return -_soft_min_score(centers, beta=beta)

def _initial_grid_444():
    """4x4x4 grid, first 50 points."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _repulsive_dynamics(centers, time_limit=5.0, steps=1000):
    """
    Repulsive dynamics to expand the packing.
    Forces: 
    1. Repulsion between spheres: F = k / d^2 * (r_i - r_j)
    2. Repulsion from walls: F = k_wall / d_wall^2 * (normal)
    """
    current = centers.copy()
    start_time = time.time()
    
    # Parameters
    k_repel = 0.01
    k_wall = 0.01
    damping = 0.9
    dt = 0.01
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        # Use KDTree for efficiency? For N=50, O(N^2) is fine (2500 pairs).
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    # Jitter if coincident
                    diff = np.random.uniform(-0.01, 0.01, 3)
                    d = np.linalg.norm(diff)
                # Force magnitude
                f_mag = k_repel / (d * d)
                f_vec = (diff / d) * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec
                
        # Wall repulsion
        for i in range(N):
            for axis in range(3):
                # Distance to positive wall
                d_pos = 0.5 - current[i, axis]
                if d_pos < 0.1: # Only if close
                    f_mag = k_wall / (d_pos * d_pos + 1e-6)
                    forces[i, axis] -= f_mag # Push inwards (negative direction)
                
                # Distance to negative wall
                d_neg = 0.5 + current[i, axis]
                if d_neg < 0.1:
                    f_mag = k_wall / (d_neg * d_neg + 1e-6)
                    forces[i, axis] += f_mag # Push inwards (positive direction)
        
        # Update positions
        current += forces * dt
        current *= damping # Damping towards center? No, just velocity damping if we had velocity.
        # Here we are doing simple gradient descent on potential energy.
        # Clip to cube
        current = np.clip(current, -0.5, 0.5)
        
    return current

def _l_bfgs_refine(centers, time_limit=10.0):
    """Refine using L-BFGS-B with soft-minimum objective."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Coarse pass
    if time.time() - start_time < time_limit * 0.5:
        result1 = minimize(
            _objective_flat, x0, args=(20.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x0 = result1.x
    
    # Fine pass
    if time.time() - start_time < time_limit * 0.9:
        result2 = minimize(
            _objective_flat, x0, args=(80.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        x0 = result2.x
    
    return np.clip(x0.reshape(N, 3), -0.5, 0.5)

def _kick_local_search(centers, time_limit=5.0, max_iter=10000):
    """
    Local search that resolves overlaps by kicking spheres apart.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find minimum distance pair
        tree = cKDTree(current)
        dists, idxs = tree.query(current, k=2)
        min_d = np.min(dists[:, 1])
        min_idx_i = np.argmin(dists[:, 1])
        min_idx_j = idxs[min_idx_i, 1]
        
        # Check wall distance
        wall_dists = 0.5 - np.abs(current)
        min_wall = np.min(wall_dists)
        
        current_radius = min(min_d / 2.0, min_wall)
        
        if current_radius > best_score:
            best = current.copy()
            best_score = current_radius
        
        # If there is an overlap (min_d < 2 * best_score), kick them apart
        if min_d < 2.0 * best_score * 1.01:
            diff = current[min_idx_i] - current[min_idx_j]
            d = np.linalg.norm(diff)
            if d > 1e-6:
                # Push apart
                push = (diff / d) * (2.0 * best_score * 1.01 - min_d) * 0.5
                current[min_idx_i] += push
                current[min_idx_j] -= push
        
        # Also handle wall collisions
        for i in range(N):
            for axis in range(3):
                if current[i, axis] > 0.5 - best_score * 1.01:
                    current[i, axis] = 0.5 - best_score * 1.01
                elif current[i, axis] < -0.5 + best_score * 1.01:
                    current[i, axis] = -0.5 + best_score * 1.01
        
        # Small random perturbation to escape local minima
        if it % 100 == 0:
            idx = np.random.randint(N)
            current[idx] += np.random.uniform(-0.005, 0.005, 3)
            current = np.clip(current, -0.5, 0.5)
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initialize with 4x4x4 grid (better initial radius than 5x5x2)
    centers = _initial_grid_444()
    best_centers = centers.copy()
    best_score = _exact_score(best_centers)
    
    # 2. Repulsive Dynamics to expand
    if time.time() < deadline - 30.0:
        dynamic_centers = _repulsive_dynamics(centers, time_limit=10.0)
        dyn_score = _exact_score(dynamic_centers)
        if dyn_score > best_score:
            best_score = dyn_score
            best_centers = dynamic_centers
    
    # 3. Kick Local Search
    if time.time() < deadline - 20.0:
        ls_centers, ls_score = _kick_local_search(best_centers, time_limit=10.0)
        if ls_score > best_score:
            best_score = ls_score
            best_centers = ls_centers
    
    # 4. L-BFGS-B Refinement
    if time.time() < deadline - 10.0:
        refined = _l_bfgs_refine(best_centers, time_limit=15.0)
        ref_score = _exact_score(refined)
        if ref_score > best_score:
            best_score = ref_score
            best_centers = refined
    
    # 5. Final polish with local search
    if time.time() < deadline - 5.0:
        final_centers, final_score = _kick_local_search(best_centers, time_limit=5.0)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers
            
    return best_centers
# EVOLVE-BLOCK-END
