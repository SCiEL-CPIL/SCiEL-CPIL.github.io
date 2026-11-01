"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score_fast(positions):
    """
    Fast calculation of the maximum radius r.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Wall distances
    min_wall = np.min(0.5 - np.abs(positions))
    
    if min_wall <= 0:
        return 0.0
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _anneal(positions, time_limit, rng_seed=0):
    """
    Simulated Annealing with Metropolis-Hastings moves.
    Energy E = -r. We minimize E (maximize r).
    """
    start_time = time.time()
    current = positions.copy()
    current_r = _score_fast(current)
    best = current.copy()
    best_r = current_r
    
    rng = np.random.default_rng(rng_seed)
    
    # Cooling schedule parameters
    T = 0.05
    alpha = 0.999
    step_size = 0.05
    
    last_improve_time = start_time
    no_improve_count = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Move one random particle
        idx = rng.integers(0, N)
        
        # Propose move
        proposed = current.copy()
        proposed[idx] += rng.normal(0, step_size, size=3)
        proposed[idx] = np.clip(proposed[idx], -0.5, 0.5)
        
        # Calculate new radius
        proposed_r = _score_fast(proposed)
        
        # Metropolis acceptance
        if proposed_r >= current_r:
            current = proposed
            current_r = proposed_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
                last_improve_time = time.time()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta_r = proposed_r - current_r
            if delta_r < 0:
                exp_arg = delta_r / T
                if exp_arg < -700:
                    prob = 0.0
                else:
                    prob = np.exp(exp_arg)
            else:
                prob = 1.0
            
            if rng.random() < prob:
                current = proposed
                current_r = proposed_r
                no_improve_count += 1
            else:
                no_improve_count += 1

        # Adaptation
        if no_improve_count > 1000:
            # If stuck, try larger moves
            step_size *= 1.1
            no_improve_count = 0
            
        # Cool down
        T *= alpha
        if T < 0.0001:
            T = 0.0001

    return best, best_r

def construct():
    """Construct positions for 100 equal spheres in a unit cube."""
    n = N
    
    # Generate multiple initial configurations
    starts = []
    
    # 1. 4x5x5 Grid (100 points exactly)
    ticks_x = (np.arange(4) + 0.5) / 4 - 0.5
    ticks_y = (np.arange(5) + 0.5) / 5 - 0.5
    ticks_z = (np.arange(5) + 0.5) / 5 - 0.5
    grid_455 = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    starts.append(grid_455)
    
    # 2. 5x5x5 Grid (125 points, pick 100)
    ticks_5 = (np.arange(5) + 0.5) / 5 - 0.5
    grid_555 = np.array([(x, y, z) for z in ticks_5 for y in ticks_5 for x in ticks_5])
    idxs = np.random.choice(len(grid_555), size=n, replace=False)
    starts.append(grid_555[idxs])
    
    # 3. Random stratified (6x6x6 grid, pick 100)
    ticks_6 = (np.arange(6) + 0.5) / 6 - 0.5
    grid_666 = np.array([(x, y, z) for z in ticks_6 for y in ticks_6 for x in ticks_6])
    idxs_6 = np.random.choice(len(grid_666), size=n, replace=False)
    starts.append(grid_666[idxs_6])
    
    best_pos = None
    best_score = -1.0
    
    # Allocate time for each start
    # Total time budget: 110s (leave 10s for safety)
    time_per_start = 30.0
    
    for i, start_pos in enumerate(starts):
        # Score initial
        init_score = _score_fast(start_pos)
        
        # Run annealing
        pos, score = _anneal(start_pos, time_per_start, rng_seed=42 + i)
        
        if score > best_score:
            best_score = score
            best_pos = pos
    
    # Final polish
    if time.time() < 110:
        pos, score = _anneal(best_pos, 5.0, rng_seed=999)
        if score > best_score:
            best_pos = pos
            best_score = score
            
    return best_pos
# EVOLVE-BLOCK-END
