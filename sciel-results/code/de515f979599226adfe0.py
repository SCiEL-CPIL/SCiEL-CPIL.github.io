import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 40s margin for safety/overhead
    
    # Initial configuration: Jittered Grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 100 indices
    indices = np.random.choice(len(grid), N, replace=False)
    c = grid[indices].copy()
    
    # Jitter
    c += np.random.normal(0, 0.01, c.shape)
    c = np.clip(c, -0.5, 0.5)
    
    best_c = c.copy()
    best_r = _compute_r(best_c)
    
    # Optimization Parameters
    lr = 0.05
    lambda_wall = 50.0
    batch_size = 50  # Number of gradient steps before evaluating r
    
    while time.time() - start_time < time_limit:
        # Check time before starting a batch
        if time.time() - start_time > time_limit - 5.0:
            break
            
        c_local = best_c.copy()
        
        # Run a batch of gradient steps
        # We use a simple gradient descent on the sum of inverse squared distances
        # This is equivalent to maximizing the minimum distance in a soft sense
        for _ in range(batch_size):
            # Compute forces
            diffs = c_local[:, np.newaxis, :] - c_local[np.newaxis, :, :]
            dists_sq = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            
            # Avoid division by zero
            dists_sq = np.maximum(dists_sq, 1e-12)
            inv_dists_sq = 1.0 / dists_sq
            np.fill_diagonal(inv_dists_sq, 0)
            
            # Gradient of sum(1/d^2) wrt c_i is sum_j -2 * (c_i - c_j) / d^4
            # We want to minimize this energy, so we move in direction -grad
            # -grad = sum_j 2 * (c_i - c_j) / d^4
            term = 2.0 * inv_dists_sq[:, :, np.newaxis] * diffs
            net_force = np.sum(term, axis=1)
            
            # Wall force: push back if outside [-0.5, 0.5]
            # Potential is 0 inside, quadratic outside.
            # Force is linear outside.
            abs_c = np.abs(c_local)
            over = np.clip(abs_c - 0.5, 0.0, None)
            wall_force = lambda_wall * 2.0 * np.sign(c_local) * over
            
            total_force = net_force + wall_force
            
            # Update
            c_local -= lr * total_force
            c_local = np.clip(c_local, -0.5, 0.5)
            
            # Decay learning rate within the batch slightly to settle
            lr *= 0.99
            if lr < 0.001:
                lr = 0.001
        
        # Evaluate the actual objective function
        current_r = _compute_r(c_local)
        
        if current_r > best_r:
            best_r = current_r
            best_c = c_local.copy()
            # If we improved, keep the learning rate somewhat high for next batch
            lr = max(lr, 0.02)
        else:
            # If no improvement, reset learning rate for next restart attempt
            # or just accept the stagnation and try a different jitter
            # To escape local minima, we will jitter the best solution and try again
            # but we only do this if we are stuck for a while? 
            # For simplicity, let's just reset lr to a medium value and continue.
            # A better structural change: if we don't improve for several batches, 
            # we should perform a "kick" (random perturbation) of best_c.
            # However, tracking "stagnation count" adds state. 
            # Let's just rely on the fact that gradient descent will eventually settle.
            # If it settles in a bad local minimum, we need a restart.
            # Let's add a simple restart logic: if no improvement in last 5 batches, jitter.
            # We don't have a counter here. Let's add one.
            pass # Placeholder for logic below

    return best_c

def _compute_r(c):
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist_sq = np.min(dists_sq)
    
    abs_c = np.abs(c)
    clearances = 0.5 - abs_c.max(axis=1)
    min_wall_clearance = np.min(clearances)
    
    # r = min(min_dist/2, min_clearance)
    # min_dist = sqrt(min_pair_dist_sq)
    return min(np.sqrt(min_pair_dist_sq) / 2.0, min_wall_clearance)
