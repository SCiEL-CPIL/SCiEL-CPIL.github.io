import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    # Initial configuration: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    # Add small random noise to break symmetry
    pos += np.random.normal(0, 0.001, pos.shape)

    # Initialize distance matrix
    # D[i, j] = distance between i and j
    # Set diagonal to inf
    D = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    
    # Current min pair dist
    min_pair_dist = np.min(D)
    
    # Wall distances
    def get_wall_dists(p):
        return 0.5 - np.max(np.abs(p), axis=1)
    
    wall_dists = get_wall_dists(pos)
    min_wall_dist = np.min(wall_dists)
    
    current_r = min(min_pair_dist / 2.0, min_wall_dist)
    best_pos = pos.copy()
    best_r = current_r

    # Simulated Annealing
    T = 0.005
    T_min = 1e-5
    alpha = 0.999
    base_step = 0.01

    steps = 0
    while time.time() - start_time < time_limit:
        i = np.random.randint(N)
        
        # Generate move
        # Scale step size with T
        step_size = base_step * (1.0 + T / 0.005)
        delta = np.random.normal(0, step_size, 2)
        
        new_pos = pos.copy()
        new_pos[i] += delta
        
        # Update D for row/col i
        # dists from i to all others
        new_dists_i = np.linalg.norm(new_pos[i] - new_pos, axis=1)
        D[i, :] = new_dists_i
        D[:, i] = new_dists_i
        D[i, i] = np.inf # Ensure diagonal is inf
        
        # Update wall dists for i
        new_wall_dist_i = 0.5 - np.max(np.abs(new_pos[i]))
        wall_dists[i] = new_wall_dist_i
        
        # Calculate new min distances
        # np.min(D) is O(N^2) but fast
        new_min_pair = np.min(D)
        new_min_wall = np.min(wall_dists)
        new_r = min(new_min_pair / 2.0, new_min_wall)
        
        # Acceptance
        if new_r > current_r:
            pos = new_pos
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pos = pos.copy()
        else:
            delta_r = current_r - new_r
            if delta_r > 0:
                prob = np.exp(-delta_r / T)
                if np.random.random() < prob:
                    pos = new_pos
                    current_r = new_r
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart logic: if stuck for a while, re-perturb
            # Simple heuristic: if T is at min, and we haven't improved best_r in last X steps?
            # Let's just restart T and add noise to current best
            if best_r < 0.036: # Far from known best
                T = 0.005
                pos = best_pos + np.random.normal(0, 0.005, best_pos.shape)
                # Recalculate D and wall dists for new pos
                D = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
                np.fill_diagonal(D, np.inf)
                wall_dists = get_wall_dists(pos)
                current_r = min(np.min(D) / 2.0, np.min(wall_dists))
        
        steps += 1
        if steps % 1000 == 0:
            # Optional: local refinement
            pass

    # Final local optimization on best_pos
    # Try to push apart the closest pairs
    final_pos = best_pos.copy()
    D_final = np.linalg.norm(final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(D_final, np.inf)
    
    for _ in range(1000):
        if time.time() - start_time > time_limit - 5:
            break
        # Find closest pair
        min_val = np.min(D_final)
        min_idx = np.unravel_index(np.argmin(D_final), D_final.shape)
        i, j = min_idx
        
        # Push apart
        vec = final_pos[i] - final_pos[j]
        norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-9:
            vec = np.array([1.0, 0.0])
            norm_vec = 1.0
        else:
            vec /= norm_vec
        
        # Move i and j apart by half the overlap?
        # Target dist is 2 * best_r
        # Just push a bit
        push = 0.001
        final_pos[i] += vec * push
        final_pos[j] -= vec * push
        
        # Check walls
        # If i or j goes out, pull back
        for idx in [i, j]:
            x, y = final_pos[idx]
            if x < -0.5: final_pos[idx][0] = -0.5
            if x > 0.5: final_pos[idx][0] = 0.5
            if y < -0.5: final_pos[idx][1] = -0.5
            if y > 0.5: final_pos[idx][1] = 0.5
            
        # Update D
        D_final[i, :] = np.linalg.norm(final_pos[i] - final_pos, axis=1)
        D_final[:, i] = D_final[i, :]
        D_final[j, :] = np.linalg.norm(final_pos[j] - final_pos, axis=1)
        D_final[:, j] = D_final[j, :]
        D_final[i, i] = np.inf
        D_final[j, j] = np.inf

    # Final score check
    D_check = np.linalg.norm(final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(D_check, np.inf)
    min_pair = np.min(D_check)
    wall_dists_final = 0.5 - np.max(np.abs(final_pos), axis=1)
    min_wall = np.min(wall_dists_final)
    final_r = min(min_pair / 2.0, min_wall)
    
    # Ensure we return the best of best_pos and final_pos
    r_best_pos = best_r
    if final_r > r_best_pos:
        return final_pos
    else:
        return best_pos
