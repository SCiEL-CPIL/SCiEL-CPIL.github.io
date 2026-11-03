"""Improved program for packing 100 equal spheres in a unit cube using stochastic local search."""
import numpy as np
import random
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _pairwise_distances(c):
    """Compute all pairwise distances for an array of shape (N, 3)."""
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    return iu, dists

def _score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    iu, dists = _pairwise_distances(c)
    r_pair = np.min(dists) / 2.0
    
    abs_c = np.abs(c)
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    r_bound = np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))
    
    return min(r_pair, r_bound)

def _initial_grid():
    """Create an initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _stochastic_local_search(c, max_iters=50000, T0=0.01, T1=1e-8, alpha=0.9995, 
                              swap_prob=0.3, perturb_scale=0.02):
    """
    Stochastic local search with swap-based moves and random perturbations.
    Directly optimizes for minimum pairwise distance by accepting uphill moves.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    start_time = time.time()
    time_limit = 110.0  # Leave buffer for other operations
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type: swap two spheres or perturb one/two spheres
        move_type = random.random()
        
        if move_type < swap_prob:
            # Swap two random spheres
            i, j = random.sample(range(N), 2)
            c_new = c.copy()
            c_new[i], c_new[j] = c_new[j].copy(), c_new[i].copy()
        else:
            # Perturb one or two random spheres
            n_move = 1 if random.random() < 0.7 else 2
            idxs = random.sample(range(N), n_move)
            c_new = c.copy()
            for idx in idxs:
                delta = np.random.randn(3) * perturb_scale
                c_new[idx] = c_new[idx] + delta
                c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            # Worse, accept with probability exp((new_score - current_score) / T)
            delta_score = new_score - current_score  # negative
            if delta_score / T > -50:  # Avoid underflow
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

def _local_refine(c, max_iters=3000):
    """
    Local refinement: push apart closest pairs or pull from boundaries.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    start_time = time.time()
    time_limit = 30.0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the limiting constraint
        iu, dists = _pairwise_distances(c)
        r_pair = np.min(dists) / 2.0
        
        abs_c = np.abs(c)
        dists_x = 0.5 - abs_c[:, 0]
        dists_y = 0.5 - abs_c[:, 1]
        dists_z = 0.5 - abs_c[:, 2]
        r_bound = np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            # Push i and j apart
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try different move sizes
            for move in [0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        else:
            # Boundary is limiting
            all_dists = np.column_stack([dists_x, dists_y, dists_z])
            min_idx = np.argmin(all_dists.flatten())
            i = min_idx // 3
            axis = min_idx % 3
            
            # Push point i away from the boundary
            sign = 1 if c[i, axis] > 0 else -1
            for move in [0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        
        # Random perturbation occasionally to escape local minima
        if it % 50 == 0:
            idx = random.randint(0, N - 1)
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
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    start_time = time.time()
    time_limit = 115.0  # Leave buffer
    
    # Start with a grid
    c = _initial_grid()
    
    # Phase 1: Broad stochastic local search
    c, score = _stochastic_local_search(c, max_iters=30000, T0=0.01, T1=1e-7, 
                                         alpha=0.9995, swap_prob=0.3, perturb_scale=0.02)
    
    if time.time() - start_time > time_limit:
        return c
    
    # Phase 2: Local refinement
    c, score2 = _local_refine(c, max_iters=2000)
    if score2 > score:
        score = score2
    
    if time.time() - start_time > time_limit:
        return c
    
    # Phase 3: Fine-tuned stochastic local search
    c, score3 = _stochastic_local_search(c, max_iters=20000, T0=0.005, T1=1e-8, 
                                          alpha=0.9997, swap_prob=0.2, perturb_scale=0.008)
    if score3 > score:
        score = score3
    
    if time.time() - start_time > time_limit:
        return c
    
    # Phase 4: Final local refinement
    c, score4 = _local_refine(c, max_iters=1500)
    if score4 > score:
        score = score4
    
    return c
# EVOLVE-BLOCK-END
