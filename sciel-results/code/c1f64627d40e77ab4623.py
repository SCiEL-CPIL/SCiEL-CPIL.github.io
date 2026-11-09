import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initial Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Precompute pair indices for efficiency
    # We need to calculate min distance between all pairs and min distance to walls.
    
    def calc_score(c):
        # c is (N, 3)
        # Pairwise distances
        # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        # This creates a 100x100x3 array, which is fine.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # min_k (0.5 - |c_ik|)
        # c is in [-0.5, 0.5] ideally
        # If c is outside, 0.5 - |c| is negative.
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    best_centers = centers.copy()
    best_score = calc_score(best_centers)
    
    # 2. Repulsion Phase (Local Relaxation)
    # This helps to break the rigid grid structure.
    # We use a simple force-directed relaxation.
    current_centers = centers.copy()
    step_size = 0.01
    decay = 0.999
    
    for i in range(2000): # Limit iterations to keep time low
        if time.time() > deadline:
            break
            
        # Calculate forces
        forces = np.zeros_like(current_centers)
        
        # Pairwise repulsion
        # We can vectorize this too, but N=100 is small enough for a loop if needed, 
        # but vectorized is better.
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Force magnitude 1/d^2
        # F_i = sum_j (c_i - c_j) / dist_ij^2
        # This is equivalent to:
        # inv_d2 = 1.0 / (dists ** 2)
        # np.fill_diagonal(inv_d2, 0.0)
        # forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        inv_d2 = 1.0 / (dists ** 2)
        np.fill_diagonal(inv_d2, 0.0)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        # Push away from walls if close
        # Simple linear spring for walls
        wall_margin = 0.05
        # If x > 0.5 - wall_margin, force is negative (push left)
        # If x < -0.5 + wall_margin, force is positive (push right)
        
        # Vectorized wall force
        # fx = 0
        # for each axis k:
        #   if c_ik > 0.5 - margin: fx -= (c_ik - (0.5 - margin))
        #   if c_ik < -0.5 + margin: fx += ((-0.5 + margin) - c_ik)
        
        # Let's just clamp and add a small inward force if on boundary?
        # Or simpler: just let the pairwise repulsion handle most of it, 
        # and add a hard constraint that centers stay in [-0.5, 0.5].
        # If we clamp, the "force" from walls is effectively infinite.
        
        # Update
        current_centers += step_size * forces
        # Clamp to cube
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        # Check score
        score = calc_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
        step_size *= decay
        
    # 3. Simulated Annealing Phase
    # Use the best found so far as start
    current_centers = best_centers.copy()
    current_score = best_score
    
    T = 0.01
    T_min = 1e-6
    alpha = 0.9999
    sigma = 0.05
    
    # Number of iterations
    # Estimate time per iteration: ~1ms?
    # We have ~100s left.
    # Let's run until deadline.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random sphere
        i = np.random.randint(N)
        
        # Generate move
        delta = np.random.normal(0, sigma, size=3)
        new_c = current_centers[i] + delta
        
        # Clamp
        new_c = np.clip(new_c, -0.5, 0.5)
        
        # Calculate new score
        # To speed up, we only need to recalculate distances involving i?
        # No, min_pair_dist could change if the new distance is smaller than the current min,
        # OR if the previous min pair involved i and the new distance is larger, the min might increase.
        # So we must recalculate global min.
        
        old_c_i = current_centers[i]
        current_centers[i] = new_c
        new_score = calc_score(current_centers)
        
        # Acceptance
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = current_centers.copy()
        else:
            # Metropolis criterion
            # We are maximizing, so delta_E = current_score - new_score (positive if worse)
            delta_E = current_score - new_score
            if delta_E < 0: # Should not happen if we checked >, but floating point
                pass
            else:
                if np.random.rand() < np.exp(-delta_E / T):
                    current_score = new_score
                else:
                    # Reject
                    current_centers[i] = old_c_i
            
        T *= alpha
        sigma *= 0.99995
        
    return best_centers
