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
    """Return the minimum pairwise distance."""
    iu, dists = _pairwise_distances(c)
    return np.min(dists)

def _min_boundary_distance(c):
    """Return the minimum distance from any centre to the nearest face of the cube [-0.5, 0.5]^3."""
    abs_c = np.abs(c)
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    return np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))

def _score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    r_pair = _min_pairwise_distance(c) / 2.0
    r_bound = _min_boundary_distance(c)
    return min(r_pair, r_bound)

def _initial_grid():
    """Create an initial grid placement with slight jitter."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N].copy()
    # Add small random jitter to break symmetry
    grid += np.random.randn(N, 3) * 0.001
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def _simulated_annealing(c, max_iters=20000, T0=0.01, T1=1e-6, alpha=0.9995, step_scale=0.02):
    """
    Simulated annealing with overlap-aware moves.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    
    for it in range(max_iters):
        # Determine move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Single point random move
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            delta = np.random.randn(3) * step_scale * (T / T0 + 0.1)
            c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        elif move_type < 0.7:
            # Two-point random move
            idxs = random.sample(range(N), 2)
            c_new = c.copy()
            for idx in idxs:
                delta = np.random.randn(3) * step_scale * (T / T0 + 0.1)
                c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        elif move_type < 0.9:
            # Overlap-aware: find closest pair and push apart
            iu, dists = _pairwise_distances(c)
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            c_new = c.copy()
            move = step_scale * 0.5 * (T / T0 + 0.1)
            c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
            c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
        else:
            # Three-point random move
            idxs = random.sample(range(N), 3)
            c_new = c.copy()
            for idx in idxs:
                delta = np.random.randn(3) * step_scale * 0.7 * (T / T0 + 0.1)
                c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            delta_score = new_score - current_score
            prob = math.exp(delta_score / T)
            if random.random() < prob:
                c = c_new
                current_score = new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Cool down
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _local_refine(c, max_iters=5000):
    """
    Local refinement focusing on bottleneck constraints.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    for it in range(max_iters):
        r_pair = _min_pairwise_distance(c) / 2.0
        r_bound = _min_boundary_distance(c)
        
        improved = False
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            iu, dists = _pairwise_distances(c)
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            for move in [0.005, 0.002, 0.001, 0.0005]:
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
            # Boundary is limiting
            abs_c = np.abs(c)
            dists_x = 0.5 - abs_c[:, 0]
            dists_y = 0.5 - abs_c[:, 1]
            dists_z = 0.5 - abs_c[:, 2]
            all_dists = np.column_stack([dists_x, dists_y, dists_z])
            min_idx = np.argmin(all_dists.flatten())
            i = min_idx // 3
            axis = min_idx % 3
            
            sign = 1 if c[i, axis] > 0 else -1
            
            for move in [0.005, 0.002, 0.001, 0.0005]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Try random perturbation to escape local minimum
            n_perturb = random.randint(1, 3)
            idxs = random.sample(range(N), n_perturb)
            c_new = c.copy()
            for idx in idxs:
                c_new[idx] = c_new[idx] + np.random.randn(3) * 0.01
                c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            
            new_score = _score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Early termination if no improvement for many iterations
        if it > 1000 and abs(current_score - best_score) < 1e-10:
            break
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Start with a jittered grid
    c = _initial_grid()
    
    # Phase 1: Coarse annealing
    c, score = _simulated_annealing(c, max_iters=15000, T0=0.008, T1=1e-7, alpha=0.9996, step_scale=0.02)
    
    # Phase 2: Local refinement
    c, score = _local_refine(c, max_iters=5000)
    
    # Phase 3: Fine annealing
    c, score2 = _simulated_annealing(c, max_iters=12000, T0=0.004, T1=1e-8, alpha=0.9998, step_scale=0.01)
    if score2 > score:
        score = score2
        c, score = _local_refine(c, max_iters=3000)
    
    # Phase 4: Very fine annealing
    c, score3 = _simulated_annealing(c, max_iters=8000, T0=0.002, T1=1e-9, alpha=0.9999, step_scale=0.005)
    if score3 > score:
        score = score3
        c, score = _local_refine(c, max_iters=2000)
    
    return c
# EVOLVE-BLOCK-END
