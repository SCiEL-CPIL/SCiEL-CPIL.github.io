# EVOLVE-BLOCK-START
import numpy as np
import time

N = 50

def _full_compute_radius(centers):
    """
    Compute the maximum valid radius for a set of centers.
    Returns r, min_pair_idx, min_boundary_idx
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    abs_centers = np.abs(centers)
    dist_to_faces = 0.5 - np.max(abs_centers, axis=1)
    min_boundary_val = np.min(dist_to_faces)
    min_boundary_idx = np.argmin(dist_to_faces)
    
    if min_pair_val < min_boundary_val:
        return min_pair_val / 2.0, min_pair_idx, None
    else:
        return min_boundary_val / 2.0, None, min_boundary_idx

def _local_compute_radius(centers, i, current_min_val, min_pair_idx, min_boundary_idx):
    """
    Fast check for moving sphere i.
    Returns new_radius, new_min_pair_idx, new_min_boundary_idx
    If i is not involved in the current min, we can do O(N).
    Else, we do O(N^2).
    """
    involved = False
    if min_pair_idx is not None:
        k, l = min_pair_idx
        if i == k or i == l:
            involved = True
    if min_boundary_idx is not None:
        if i == min_boundary_idx:
            involved = True
            
    if not involved:
        # O(N) check
        diffs = centers - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf
        local_pair_min = np.min(dists)
        
        abs_c = np.abs(centers[i])
        local_bound_min = 0.5 - np.max(abs_c)
        
        local_min_val = min(local_pair_min, local_bound_min)
        
        new_radius = min(current_min_val, local_min_val) / 2.0
        
        if local_min_val < current_min_val:
            if local_pair_min < local_bound_min:
                min_j = np.argmin(dists)
                return new_radius, (i, min_j), None
            else:
                return new_radius, None, i
        else:
            return new_radius, min_pair_idx, min_boundary_idx
            
    else:
        return _full_compute_radius(centers)

def _sa_optimize(initial_centers, time_limit=100.0):
    """
    Simulated Annealing with lazy evaluation.
    """
    centers = initial_centers.copy()
    current_radius, min_pair_idx, min_boundary_idx = _full_compute_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    T = 0.05
    T_min = 1e-6
    alpha = 0.9995
    
    rng = np.random.RandomState(42)
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        step_size = 0.05 * (T / 0.05)
        
        i = rng.randint(N)
        move = rng.randn(3) * step_size
        
        new_centers = centers.copy()
        new_centers[i] += move
        
        new_radius, new_min_pair_idx, new_min_boundary_idx = _local_compute_radius(
            new_centers, i, current_radius, min_pair_idx, min_boundary_idx
        )
        
        delta_E = current_radius - new_radius
        
        if delta_E < 0:
            centers = new_centers
            current_radius = new_radius
            min_pair_idx = new_min_pair_idx
            min_boundary_idx = new_min_boundary_idx
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            prob = np.exp(-delta_E / T)
            if rng.random() < prob:
                centers = new_centers
                current_radius = new_radius
                min_pair_idx = new_min_pair_idx
                min_boundary_idx = new_min_boundary_idx
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_centers

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_centers_overall = None
    best_radius_overall = 0.0
    
    rng_global = np.random.RandomState(0)
    
    n_runs = 20
    time_per_run = time_limit / n_runs
    
    for i in range(n_runs):
        centers = rng_global.uniform(-0.45, 0.45, (N, 3))
        
        optimized_centers = _sa_optimize(centers, time_limit=time_per_run)
        radius, _, _ = _full_compute_radius(optimized_centers)
        
        if radius > best_radius_overall:
            best_radius_overall = radius
            best_centers_overall = optimized_centers.copy()
            
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return best_centers_overall
# EVOLVE-BLOCK-END
