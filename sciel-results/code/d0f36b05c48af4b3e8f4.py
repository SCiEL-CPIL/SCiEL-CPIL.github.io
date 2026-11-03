"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.optimize import minimize

def _compute_distances(c):
    """Compute all pairwise distances and boundary distances."""
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1))
    
    abs_c = np.abs(c)
    bound_dists = np.minimum.reduce([0.5 - abs_c[:, 0], 0.5 - abs_c[:, 1], 0.5 - abs_c[:, 2]])
    
    return iu, pair_dists, bound_dists

def _score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    iu, pair_dists, bound_dists = _compute_distances(c)
    r_pair = np.min(pair_dists) / 2.0
    r_bound = np.min(bound_dists)
    return min(r_pair, r_bound)

def _initial_grid_554():
    """Create a 5x5x4 grid initial placement (100 points)."""
    # 5 * 5 * 4 = 100
    x_ticks = (np.arange(5) + 0.5) / 5 - 0.5
    y_ticks = (np.arange(5) + 0.5) / 5 - 0.5
    z_ticks = (np.arange(4) + 0.5) / 4 - 0.5
    
    grid = []
    for z in z_ticks:
        for y in y_ticks:
            for x in x_ticks:
                grid.append([x, y, z])
    
    return np.array(grid)

def _simulated_annealing_v2(c, max_iters=30000, T0=0.008, T1=1e-8, alpha=0.9998):
    """
    Improved simulated annealing with overlap-aware moves.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        move_type = rng.random()
        
        c_new = c.copy()
        
        if move_type < 0.4:
            # Move one random point
            idx = rng.randint(0, N - 1)
            delta = rng.randn(3) * 0.02
            c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
            
        elif move_type < 0.7:
            # Move two random points
            idxs = rng.choice(N, size=2, replace=False)
            for idx in idxs:
                delta = rng.randn(3) * 0.015
                c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
                
        elif move_type < 0.85:
            # Find closest pair and push them apart
            iu, pair_dists, bound_dists = _compute_distances(c)
            min_idx = np.argmin(pair_dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = rng.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            move = rng.uniform(0.002, 0.01)
            c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
            c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
            
        else:
            # Swap two random points (helps escape local minima)
            idxs = rng.choice(N, size=2, replace=False)
            c_new[idxs[0]], c_new[idxs[1]] = c[idxs[1]].copy(), c[idxs[0]].copy()
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            delta_score = new_score - current_score
            if T > 1e-10:
                prob = math.exp(delta_score / T)
                if rng.random() < prob:
                    c = c_new
                    current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        T *= alpha
        if T < T1:
            break
    
    return best_c, best_score

def _local_optimize(c, max_iters=10000):
    """
    Local optimization: iteratively fix the most violated constraint.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    for it in range(max_iters):
        iu, pair_dists, bound_dists = _compute_distances(c)
        r_pair = np.min(pair_dists) / 2.0
        r_bound = np.min(bound_dists)
        
        improved = False
        
        if r_pair <= r_bound:
            # Pairwise constraint is limiting
            min_idx = np.argmin(pair_dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try multiple move sizes
            for move in [0.008, 0.004, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
        
        else:
            # Boundary constraint is limiting
            min_bound_idx = np.argmin(bound_dists)
            i = min_bound_idx
            axis = np.argmin([0.5 - abs(c[i, 0]), 0.5 - abs(c[i, 1]), 0.5 - abs(c[i, 2])])
            
            sign = 1 if c[i, axis] > 0 else -1
            for move in [0.008, 0.004, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Occasionally add random perturbation to escape local minima
        if not improved and it % 50 == 0:
            idx = np.random.randint(0, N)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.01
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
        
        # Early stopping if stuck
        if not improved and it % 200 == 199:
            # Try a bigger kick
            idx = np.random.randint(0, N)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.03
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
    
    return best_c, best_score

def _nelder_mead_refine(c, maxfev=2000):
    """
    Use Nelder-Mead to polish the solution.
    """
    def objective(x):
        c_tmp = x.reshape(N, 3)
        c_tmp = np.clip(c_tmp, -0.5, 0.5)
        return -_score(c_tmp)
    
    try:
        result = minimize(objective, c.flatten(), method='Nelder-Mead', 
                          options={'maxiter': maxfev, 'xatol': 1e-8, 'fatol': 1e-10})
        c_refined = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
        score_refined = _score(c_refined)
        
        if score_refined > _score(c):
            return c_refined, score_refined
    except Exception:
        pass
    
    return c, _score(c)

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Start with 5x5x4 grid
    c = _initial_grid_554()
    initial_score = _score(c)
    
    # First round of simulated annealing
    c, score = _simulated_annealing_v2(c, max_iters=25000, T0=0.01, T1=1e-8, alpha=0.9997)
    
    # Local optimization
    c, score = _local_optimize(c, max_iters=8000)
    
    # Second round of simulated annealing with lower temperature
    c, score2 = _simulated_annealing_v2(c, max_iters=15000, T0=0.005, T1=1e-9, alpha=0.99985)
    if score2 > score:
        score = score2
        c, score = _local_optimize(c, max_iters=5000)
    
    # Third round: fine-tuning
    c, score3 = _simulated_annealing_v2(c, max_iters=10000, T0=0.002, T1=1e-10, alpha=0.9999)
    if score3 > score:
        score = score3
        c, score = _local_optimize(c, max_iters=3000)
    
    # Final Nelder-Mead polish
    c, score = _nelder_mead_refine(c, maxfev=3000)
    
    # One more local optimization after Nelder-Mead
    c, score = _local_optimize(c, max_iters=2000)
    
    return c
# EVOLVE-BLOCK-END
