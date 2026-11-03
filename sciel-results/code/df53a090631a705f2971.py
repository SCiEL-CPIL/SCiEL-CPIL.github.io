"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def compute_radius(centers):
    """
    Compute the maximum radius for a given set of centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_{i,k} (0.5 - |c_ik|) )
    """
    if not np.all(np.isfinite(centers)):
        return -1.0
        
    # Calculate pairwise distances
    # diff shape (N, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    
    # Calculate distance to boundaries
    abs_centers = np.abs(centers)
    dists_to_boundary = 0.5 - abs_centers
    min_boundary_dist = np.min(dists_to_boundary)
    
    # The radius is the minimum of the half-pair-distance and the boundary distance
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def construct():
    """
    Use Simulated Annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    # Time management
    start_time = time.time()
    time_limit = 100.0  # Leave 20s buffer for overhead
    
    # State
    best_centers = None
    best_radius = -1.0
    
    # Parameters
    T_initial = 0.02
    T_final = 1e-6
    max_iter_per_restart = 50000
    num_restarts = 20
    
    # Calculate cooling alpha for each restart
    alpha = (T_final / T_initial) ** (1.0 / max_iter_per_restart)
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization
        # Use a jittered grid to ensure good initial spread
        # 4x4x4 grid = 64 points. Pick 50.
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        # Shuffle and pick N
        indices = rng.permutation(len(grid_points))
        centers = grid_points[indices[:N]].copy()
        
        # Add small noise to break symmetry
        centers += rng.normal(0, 0.005, size=centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        current_radius = compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
        T = T_initial
        no_improve_counter = 0
        max_no_improve = 5000
        
        for i in range(max_iter_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Select random sphere
            idx = rng.integers(0, N)
            
            # Random move
            # Step size scales with T
            step_size = 0.01 + 0.01 * (T / T_initial)
            delta = rng.normal(0, step_size, size=3)
            
            old_pos = centers[idx].copy()
            centers[idx] += delta
            
            # Keep inside cube
            # If it goes outside, radius becomes negative.
            # We can clip to [-0.5, 0.5] to avoid huge negative values, 
            # but the penalty from negative radius is strong enough.
            
            new_radius = compute_radius(centers)
            
            # Acceptance
            if new_radius > current_radius:
                current_radius = new_radius
                no_improve_counter = 0
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = centers.copy()
            else:
                # Probabilistic acceptance
                delta_r = new_radius - current_radius
                # If T is very small, exp(delta_r/T) is 0 for negative delta_r
                if delta_r < 0 and T > 1e-9:
                    prob = math.exp(delta_r / T)
                    if rng.random() < prob:
                        current_radius = new_radius
                    else:
                        # Reject
                        centers[idx] = old_pos
                else:
                    # Reject
                    centers[idx] = old_pos
                no_improve_counter += 1
            
            # Cool down
            T *= alpha
            
            if no_improve_counter > max_no_improve:
                break

    return best_centers
# EVOLVE-BLOCK-END
