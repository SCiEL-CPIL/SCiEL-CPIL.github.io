"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Simulated annealing to maximize the packing radius of 50 equal circles
    in the unit square [-1/2, 1/2]^2.
    """
    rng = np.random.default_rng(2024)
    start_time = time.time()
    max_time = 115.0

    def compute_score(centers):
        """Compute the packing radius."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair = np.sqrt(np.min(dists_sq))
        return min(min_pair / 2.0, min_wall)

    def sa_run(centers_init, time_budget):
        """Run simulated annealing with efficient incremental scoring."""
        centers = centers_init.copy()
        t0 = time.time()
        
        # Precompute wall distances
        wall_dists = 0.5 - np.abs(centers)  # shape (N, 2)
        
        # Precompute distance matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Track the minimum pair distance and which pair achieves it
        # Also track the minimum wall distance and which circle/wall achieves it
        
        min_pair_sq = np.min(dists_sq)
        min_wall = np.min(wall_dists)
        current_score = min(np.sqrt(min_pair_sq) / 2.0, min_wall)
        
        best_centers = centers.copy()
        best_score = current_score
        
        # Find which pair has the minimum distance
        min_pair_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
        min_i, min_j = min_pair_idx
        
        # Find which circle/wall has the minimum wall distance
        min_wall_flat_idx = np.argmin(wall_dists)
        min_wall_i = min_wall_flat_idx // 2
        min_wall_coord = min_wall_flat_idx % 2  # 0=x, 1=y
        
        # SA parameters
        T_start = 0.02
        T_end = 1e-10
        n_iters = 2000000
        alpha = (T_end / T_start) ** (1.0 / n_iters)
        
        step = 0.015
        base_step = 0.015
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(n_iters):
            if it % 1000 == 0:
                if time.time() - t0 > time_budget:
                    break
            
            T = T_start * (alpha ** it)
            
            # Pick random circle to move
            i = rng.integers(N)
            old_pos = centers[i].copy()
            old_wall_i = wall_dists[i].copy()
            
            # Generate new position
            new_pos = old_pos + rng.normal(0, step, 2)
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            
            # Compute new distances from i to all others
            new_dists_sq_i = np.sum((centers - new_pos) ** 2, axis=1)
            new_dists_sq_i[i] = np.inf
            
            # New wall distance for circle i
            new_wall_i = 0.5 - np.abs(new_pos)
            
            # Compute new minimum pair distance
            # The new min pair is either:
            # 1. The old min pair (if it doesn't involve i)
            # 2. A new pair involving i
            
            if min_i == i or min_j == i:
                # Old min pair involved i, so we need to find the next min pair not involving i
                # Create a mask to exclude row/col i
                mask = np.ones((N, N), dtype=bool)
                mask[i, :] = False
                mask[:, i] = False
                # Set diagonal to inf
                np.fill_diagonal(mask, False)
                
                # Get the minimum from the masked matrix
                masked_dists = np.where(mask, dists_sq, np.inf)
                old_min_excl_i = np.min(masked_dists)
                
                # New min pair is min(old_min_excl_i, min(new_dists_sq_i))
                new_min_pair_sq = min(old_min_excl_i, np.min(new_dists_sq_i))
                
                # Update which pair is the minimum
                if np.min(new_dists_sq_i) < old_min_excl_i:
                    new_min_j = np.argmin(new_dists_sq_i)
                    min_i, min_j = i, new_min_j
                else:
                    # Find the pair that achieves old_min_excl_i
                    min_pair_idx_excl = np.unravel_index(np.argmin(masked_dists), masked_dists.shape)
                    min_i, min_j = min_pair_idx_excl
            else:
                # Old min pair doesn't involve i
                # New min pair is min(old min pair, min(new_dists_sq_i))
                old_min_val = dists_sq[min_i, min_j]
                new_min_inv_i = np.min(new_dists_sq_i)
                
                if new_min_inv_i < old_min_val:
                    new_min_pair_sq = new_min_inv_i
                    new_min_j = np.argmin(new_dists_sq_i)
                    min_i, min_j = i, new_min_j
                else:
                    new_min_pair_sq = old_min_val
            
            # Compute new minimum wall distance
            if min_wall_i == i:
                # Old min wall was for circle i
                # Find the next min wall not for circle i
                wall_excl_i = np.concatenate([wall_dists[:i], wall_dists[i+1:]]).min() if N > 1 else np.inf
                new_min_wall = min(np.min(new_wall_i), wall_excl_i)
                
                # Update which circle/wall has the minimum
                if np.min(new_wall_i) < wall_excl_i:
                    min_wall_i = i
                    min_wall_coord = np.argmin(new_wall_i)
                else:
                    # Find the circle/wall that achieves wall_excl_i
                    wall_flat = np.concatenate([wall_dists[:i].flatten(), wall_dists[i+1:].flatten()])
                    min_flat_idx = np.argmin(wall_flat)
                    if min_flat_idx < i * 2:
                        min_wall_i = min_flat_idx // 2
                        min_wall_coord = min_flat_idx % 2
                    else:
                        adjusted_idx = min_flat_idx - i * 2
                        min_wall_i = adjusted_idx // 2 + i + 1
                        min_wall_coord = adjusted_idx % 2
            else:
                # Old min wall was for another circle
                old_min_wall_val = wall_dists[min_wall_i, min_wall_coord]
                new_min_wall_i_val = np.min(new_wall_i)
                
                if new_min_wall_i_val < old_min_wall_val:
                    new_min_wall = new_min_wall_i_val
                    min_wall_i = i
                    min_wall_coord = np.argmin(new_wall_i)
                else:
                    new_min_wall = old_min_wall_val
            
            # Compute new score
            new_score = min(np.sqrt(new_min_pair_sq) / 2.0, new_min_wall)
            
            # Decide whether to accept the move
            delta = new_score - current_score
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-15)):
                # Accept the move
                centers[i] = new_pos
                wall_dists[i] = new_wall_i
                
                # Update distance matrix
                diff_i = centers - new_pos
                new_dists_row = np.sum(diff_i ** 2, axis=1)
                new_dists_row[i] = np.inf
                dists_sq[i, :] = new_dists_row
                dists_sq[:, i] = new_dists_row
                
                min_pair_sq = new_min_pair_sq
                min_wall = new_min_wall
                current_score = new_score
                no_improve = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = centers.copy()
            else:
                no_improve += 1
            
            # Adaptive step size
            if no_improve > max_no_improve:
                step *= 0.7
                no_improve = 0
            elif it % 5000 == 0 and it > 0:
                step *= 0.995
            
            # Periodically increase step to escape local minima
            if it % 50000 == 0 and it > 0:
                step = base_step * (0.8 + 0.4 * rng.random())
                no_improve = 0
        
        # Final local refinement
        remaining = time_budget - (time.time() - t0)
        if remaining > 2.0:
            centers = best_centers.copy()
            refine_budget = min(10.0, remaining)
            refine_start = time.time()
            refine_step = 0.003
            cur_score = compute_score(centers)
            
            while time.time() - refine_start < refine_budget:
                improved = False
                for _ in range(200):
                    i = rng.integers(N)
                    old_pos = centers[i].copy()
                    new_pos = old_pos + rng.normal(0, refine_step, 2)
                    new_pos = np.clip(new_pos, -0.4999, 0.4999)
                    centers[i] = new_pos
                    new_score = compute_score(centers)
                    
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_centers = centers.copy()
                    else:
                        centers[i] = old_pos
                
                if not improved:
                    refine_step *= 0.5
                    if refine_step < 1e-7:
                        break
        
        return best_centers, best_score

    # Generate diverse initializations
    initializations = []
    
    # 1. Grid-based initialization
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]:
        x_ticks = np.linspace(-0.48, 0.48, cols)
        y_ticks = np.linspace(-0.48, 0.48, rows)
        centers = np.zeros((N, 2))
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < N:
                    centers[idx] = [x_ticks[c], y_ticks[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
        centers += rng.normal(0, 0.005, (N, 2))
        centers = np.clip(centers, -0.4999, 0.4999)
        initializations.append(centers)
    
    # 2. Hexagonal packing
    hex_centers = np.zeros((N, 2))
    hex_spacing = 0.185
    hex_row_h = hex_spacing * np.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < N and row < 15:
        offset = 0 if row % 2 == 0 else hex_spacing / 2
        col = 0
        while idx < N and col < 12:
            x = -0.47 + offset + col * hex_spacing
            y = -0.47 + row * hex_row_h
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                hex_centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    if idx == N:
        hex_centers += rng.normal(0, 0.005, (N, 2))
        hex_centers = np.clip(hex_centers, -0.4999, 0.4999)
        initializations.append(hex_centers)
    
    # 3. Random initialization
    initializations.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Another random initialization with different seed
    initializations.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # Run SA from each initialization
    best_centers = None
    best_score = -np.inf
    
    n_inits = len(initializations)
    for init_idx, init_centers in enumerate(initializations):
        remaining = max_time - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        # Allocate time budget
        time_budget = min(remaining, max(12.0, remaining / (n_inits - init_idx)))
        
        centers_local, score_local = sa_run(init_centers, time_budget)
        
        if score_local > best_score:
            best_score = score_local
            best_centers = centers_local
    
    # Fallback if something went wrong
    if best_centers is None:
        x_ticks = np.linspace(-0.48, 0.48, 8)
        y_ticks = np.linspace(-0.48, 0.48, 7)
        best_centers = np.zeros((N, 2))
        idx = 0
        for r in range(7):
            for c in range(8):
                if idx < N:
                    best_centers[idx] = [x_ticks[c], y_ticks[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
    
    # Final verification and cleanup
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
# EVOLVE-BLOCK-END
