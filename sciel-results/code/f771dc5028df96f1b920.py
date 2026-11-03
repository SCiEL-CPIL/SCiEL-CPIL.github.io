"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def _pairwise_distances(c):
    """Compute all pairwise distances for an array of shape (N, 3)."""
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    return iu, dists

def _min_pairwise_distance(c):
    """Return the minimum pairwise distance and the indices of the pair."""
    iu, dists = _pairwise_distances(c)
    min_idx = np.argmin(dists)
    return dists[min_idx], iu[0][min_idx], iu[1][min_idx]

def _min_boundary_distance(c):
    """Return the minimum distance from any centre to the nearest face of the cube [-0.5, 0.5]^3."""
    abs_c = np.abs(c)
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    all_dists = np.column_stack([dists_x, dists_y, dists_z])
    min_idx = np.argmin(all_dists.flatten())
    i = min_idx // 3
    axis = min_idx % 3
    return all_dists.flatten()[min_idx], i, axis

def _score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    r_pair, _, _ = _min_pairwise_distance(c)
    r_bound, _, _ = _min_boundary_distance(c)
    return min(r_pair / 2.0, r_bound)

def _initial_grid():
    """Create an initial grid placement."""
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _clamp_to_cube(c):
    """Clamp centres to be inside the cube [-0.5, 0.5]^3."""
    return np.clip(c, -0.5, 0.5)

def _stochastic_local_search(c, max_iters=50000, T0=0.01, T1=1e-5, alpha=0.999):
    """
    Improved stochastic local search with critical pair moves.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    
    for it in range(max_iters):
        # Decide move type
        move_type = random.random()
        
        c_new = c.copy()
        
        if move_type < 0.6:
            # Critical pair move: push closest pair apart
            _, i, j = _min_pairwise_distance(c)
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Random step size
            step = random.uniform(0.005, 0.02)
            c_new[i] += direction * step
            c_new[j] -= direction * step
            
        elif move_type < 0.8:
            # Boundary move: push boundary-constrained point inwards
            _, i, axis = _min_boundary_distance(c)
            sign = 1 if c[i, axis] > 0 else -1
            step = random.uniform(0.005, 0.02)
            c_new[i, axis] += sign * step
            
        else:
            # Random move
            idx = random.randint(0, N - 1)
            c_new[idx] += np.random.randn(3) * 0.02
        
        c_new = _clamp_to_cube(c_new)
        new_score = _score(c_new)
        
        # Acceptance
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            delta = new_score - current_score
            if random.random() < math.exp(delta / T):
                c = c_new
                current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        T *= alpha
        if T < T1:
            break
            
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Start with a grid
    c = _initial_grid()
    
    # Run stochastic local search
    c, score = _stochastic_local_search(c, max_iters=30000, T0=0.01, T1=1e-5, alpha=0.9995)
    
    # Second round with lower temperature
    c, score2 = _stochastic_local_search(c, max_iters=20000, T0=0.005, T1=1e-6, alpha=0.9998)
    if score2 > score:
        c, score = c, score2
        
    return c
# EVOLVE-BLOCK-END
