"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

def _pack_radius(c, tree=None, k=6):
    """Fast approximate packing radius using KDTree nearest neighbors."""
    if tree is None:
        tree = cKDTree(c)
    # wall distance
    wall = np.min(0.5 - np.abs(c), axis=1)
    # nearest neighbor distance (excluding self)
    if k < 2:
        k = 2
    d, _ = tree.query(c, k=k)
    # d[:, 0] is 0 (self), d[:, 1] is nearest neighbor
    nn_dist = d[:, 1]
    return np.minimum(np.min(wall), np.min(nn_dist) / 2.0)

def _anneal(c, time_limit, k=6):
    """Simulated annealing with random swaps and perturbations."""
    c = c.copy()
    start_time = time.time()
    
    # Initial temperature based on current radius
    r0 = _score(c)
    T = 0.001
    T_min = 1e-6
    cooling = 0.999
    
    best_c = c.copy()
    best_r = r0
    
    tree = cKDTree(c)
    current_r = r0
    
    while time.time() < time_limit:
        # Choose move type
        move_type = np.random.random()
        
        if move_type < 0.4:
            # Swap two points
            i, j = np.random.choice(N, 2, replace=False)
            new_c = c.copy()
            new_c[[i, j]] = new_c[[j, i]]
        elif move_type < 0.7:
            # Perturb one point
            i = np.random.randint(N)
            delta = np.random.normal(0, T * 2, size=3)
            new_c = c.copy()
            new_c[i] = np.clip(c[i] + delta, -0.4999, 0.4999)
        else:
            # Perturb two points
            i, j = np.random.choice(N, 2, replace=False)
            delta_i = np.random.normal(0, T * 2, size=3)
            delta_j = np.random.normal(0, T * 2, size=3)
            new_c = c.copy()
            new_c[i] = np.clip(c[i] + delta_i, -0.4999, 0.4999)
            new_c[j] = np.clip(c[j] + delta_j, -0.4999, 0.4999)
        
        # Evaluate new configuration
        new_r = _pack_radius(new_c, k=k)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = current_r - new_r
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Periodically rebuild tree for accuracy
        if np.random.random() < 0.01:
            tree = cKDTree(c)
            current_r = _pack_radius(c, tree, k)
    
    return best_c, best_r

def _local_refine(c, time_limit):
    """Refine using local search on nearest neighbors."""
    c = c.copy()
    start_time = time.time()
    r = _score(c)
    best_c = c.copy()
    best_r = r
    
    for _ in range(1000):
        if time.time() > time_limit:
            break
        
        # Find the most constrained pair (closest pair)
        tree = cKDTree(c)
        d, idx = tree.query(c, k=2)
        nn_dist = d[:, 1]
        nn_idx = idx[:, 1]
        
        # Find the closest pair
        min_idx = np.argmin(nn_dist)
        i, j = min_idx, nn_idx[min_idx]
        
        # Try to move them apart
        diff = c[i] - c[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            continue
        direction = diff / dist
        
        # Move i and j apart slightly
        step = 0.001
        new_c = c.copy()
        new_c[i] = np.clip(c[i] + step * direction, -0.4999, 0.4999)
        new_c[j] = np.clip(c[j] - step * direction, -0.4999, 0.4999)
        
        new_r = _score(new_c)
        if new_r > r:
            c = new_c
            r = new_r
            if r > best_r:
                best_r = r
                best_c = c.copy()
        else:
            # Try moving just one point
            for idx in [i, j]:
                new_c = c.copy()
                new_c[idx] = np.clip(c[idx] + step * direction, -0.4999, 0.4999)
                new_r = _score(new_c)
                if new_r > r:
                    c = new_c
                    r = new_r
                    if r > best_r:
                        best_r = r
                        best_c = c.copy()
                    break
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Multiple random starts with annealing
    best_c = _init_grid().copy()
    best_r = _score(best_c)
    
    for restart in range(8):
        if time.time() > deadline - 15.0:
            break
        
        # Different initializations
        if restart == 0:
            c0 = _init_grid().copy()
        else:
            c0 = _init_random()
        
        # Anneal
        c_annealed, r_annealed = _anneal(c0, min(deadline - 20.0, start_time + 15.0 + restart * 10))
        
        if r_annealed > best_r:
            best_r = r_annealed
            best_c = c_annealed.copy()
        
        # Local refine
        if time.time() < deadline - 5.0:
            c_refined, r_refined = _local_refine(c_annealed, deadline - 5.0)
            if r_refined > best_r:
                best_r = r_refined
                best_c = c_refined.copy()
    
    # Final refinement
    if time.time() < deadline - 2.0:
        c_final, r_final = _local_refine(best_c, deadline - 2.0)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    return best_c
# EVOLVE-BLOCK-END
