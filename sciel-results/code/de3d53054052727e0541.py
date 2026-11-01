"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score_fast(centers, i):
    """
    Compute the packing radius assuming only center i has changed.
    Returns (min_pairwise_dist/2, min_wall_dist, min_total).
    """
    # Wall distance for point i
    wall_i = 0.5 - np.max(np.abs(centers[i]))
    
    # Pairwise distances from point i to all others
    diffs = centers - centers[i]
    diffs[i] = np.inf  # Exclude self
    dists_sq = np.sum(diffs * diffs, axis=1)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    
    return min_pair / 2.0, wall_i, min(min_pair / 2.0, wall_i)


def _compute_full_score(centers):
    """
    Compute the full packing radius for validation.
    """
    min_wall = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    return min(min_pair / 2.0, min_wall)


def _sa_run(centers_init, rng, start_time, max_time, n_iterations=200000):
    """
    Run simulated annealing from initial centers.
    Uses incremental score computation for speed.
    """
    centers = centers_init.copy()
    
    # Compute initial full score
    current_pair_half = None
    current_wall = None
    current_score = _compute_full_score(centers)
    
    # Cache: min pairwise distance (half) and min wall distance
    # We'll track these incrementally
    min_pair_half = current_score
    min_wall = current_score
    
    # Recompute properly
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_half = np.sqrt(np.min(dists_sq)) / 2.0
    min_wall = np.min(0.5 - np.abs(centers))
    current_score = min(min_pair_half, min_wall)
    
    best_centers = centers.copy()
    best_score = current_score
    
    T_start = 0.05
    T_end = 1e-12
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    base_step = 0.05
    step_size = base_step
    
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(n_iterations):
        if it % 1000 == 0:
            if time.time() - start_time > max_time:
                break
        
        T = T_start * (alpha ** it)
        
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Perturb
        centers[i] += rng.normal(0, step_size, 2)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        # Compute new score incrementally
        # Wall distance for point i
        new_wall_i = 0.5 - np.max(np.abs(centers[i]))
        
        # Pairwise distances from i to all others
        diffs = centers - centers[i]
        diffs[i] = np.inf
        dists_sq = np.sum(diffs * diffs, axis=1)
        new_min_pair_sq_i = np.min(dists_sq)
        new_min_pair_half_i = np.sqrt(new_min_pair_sq_i) / 2.0
        
        # The new global min is min(old min excluding i's contribution, new i's contribution)
        # But we need to be careful: the old min_pair_half might have been determined by i
        # Simple approach: recompute if i was involved in the bottleneck
        # For simplicity and correctness, let's just track:
        # new_score = min(new_wall_i, new_min_pair_half_i, other_constraints)
        # But other_constraints might be smaller... 
        
        # Efficient approach: maintain running min
        # Actually, let's just compute the new candidate score as:
        # The minimum of:
        # 1. new wall distance for point i
        # 2. new min pairwise distance involving point i  
        # 3. the previous min that didn't involve point i
        
        # To keep it simple and correct, let's do a partial recompute:
        # Check if the new position improves the bottleneck
        new_candidate = min(new_wall_i, new_min_pair_half_i)
        
        # But we also need to ensure other points still satisfy constraints
        # The true new score is min over all pairs and walls
        # Since only point i moved, the new score is:
        # min( min over j!=i of dist(i,j)/2, min over all walls )
        # = min( new_min_pair_half_i, min(new_wall_i, min over j!=i of wall_j) )
        
        # We can approximate: if new_candidate > current_score, it's definitely an improvement
        # If new_candidate < current_score, it might still be ok if the old bottleneck was i
        
        # Let's just compute the new score properly but efficiently:
        # Only need to check pairs involving i and all walls
        # Walls: min(new_wall_i, min over j!=i of (0.5 - max(|centers[j]|)))
        # We can precompute min_wall_excluding_i if needed, but for now:
        
        # Quick check: if new_candidate >= current_score, accept
        # Otherwise, we need to verify
        if new_candidate >= current_score:
            new_score = new_candidate
            # Verify by checking all walls
            all_walls = 0.5 - np.abs(centers)
            actual_min_wall = np.min(all_walls)
            new_score = min(new_min_pair_half_i, actual_min_wall)
        else:
            # Need to check if the old bottleneck was point i
            # Compute the min pairwise distance not involving i
            # This is expensive, so let's just compute full score if close
            if current_score - new_candidate < 0.001:
                # Close call, compute full score
                diffs_full = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists_sq_full = np.sum(diffs_full * diffs_full, axis=2)
                np.fill_diagonal(dists_sq_full, np.inf)
                min_pair_half_full = np.sqrt(np.min(dists_sq_full)) / 2.0
                min_wall_full = np.min(0.5 - np.abs(centers))
                new_score = min(min_pair_half_full, min_wall_full)
            else:
                new_score = new_candidate
        
        if new_score >= current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            delta = new_score - current_score
            if delta / T > -100:
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    no_improve = 0
                else:
                    centers[i] = old_pos
                    no_improve += 1
            else:
                centers[i] = old_pos
                no_improve += 1
        
        # Adaptive step size
        if no_improve > max_no_improve:
            step_size *= 0.5
            no_improve = 0
        elif it % 5000 == 0 and it > 0:
            step_size *= 0.98
        
        # Occasionally boost step size to escape local minima
        if it % 30000 == 0 and it > 0:
            step_size = base_step * (0.5 + 0.5 * rng.random())
            no_improve = 0
    
    return best_centers, best_score


def construct():
    """
    Use simulated annealing to maximize the packing radius.
    Multiple restarts with different initializations, then final refinement.
    """
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    max_time = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # 1. 7x8 grid
    rows, cols = 7, 8
    x_ticks = np.linspace(-0.48, 0.48, cols)
    y_ticks = np.linspace(-0.48, 0.48, rows)
    centers_grid = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers_grid[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    centers_grid += rng.normal(0, 0.02, (N, 2))
    centers_grid = np.clip(centers_grid, -0.4999, 0.4999)
    initializations.append(centers_grid)
    
    # 2. 5x10 grid
    rows, cols = 5, 10
    x_ticks = np.linspace(-0.48, 0.48, cols)
    y_ticks = np.linspace(-0.48, 0.48, rows)
    centers_grid2 = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers_grid2[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    centers_grid2 += rng.normal(0, 0.02, (N, 2))
    centers_grid2 = np.clip(centers_grid2, -0.4999, 0.4999)
    initializations.append(centers_grid2)
    
    # 3. 6x9 grid (54 cells, use 50)
    rows, cols = 6, 9
    x_ticks = np.linspace(-0.48, 0.48, cols)
    y_ticks = np.linspace(-0.48, 0.48, rows)
    centers_grid3 = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers_grid3[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    centers_grid3 += rng.normal(0, 0.02, (N, 2))
    centers_grid3 = np.clip(centers_grid3, -0.4999, 0.4999)
    initializations.append(centers_grid3)
    
    # 4. Random
    centers_random = rng.uniform(-0.45, 0.45, (N, 2))
    initializations.append(centers_random)
    
    # 5. Hexagonal-like pattern
    centers_hex = np.zeros((N, 2))
    row_h = 0.12
    col_w = 0.11
    idx = 0
    for r in range(8):
        y = -0.44 + r * row_h
        offset = (r % 2) * col_w / 2
        for c in range(7):
            if idx < N:
                x = -0.44 + c * col_w + offset
                centers_hex[idx] = [x, y]
                idx += 1
            else:
                break
        if idx >= N:
            break
    centers_hex += rng.normal(0, 0.015, (N, 2))
    centers_hex = np.clip(centers_hex, -0.4999, 0.4999)
    initializations.append(centers_hex)
    
    # Run SA from each initialization
    for init_idx, init_centers in enumerate(initializations):
        if time.time() - start_time > max_time - 20:
            break
        
        # Time budget for this run
        time_left = max_time - (time.time() - start_time)
        if time_left < 5:
            break
        
        # Allocate time: more time for earlier runs
        if init_idx < len(initializations) // 2:
            n_iter = 150000
        else:
            n_iter = 100000
        
        centers_local, score_local = _sa_run(
            init_centers, rng, start_time, max_time - (time.time() - start_time), n_iter
        )
        
        # Validate
        score_check = _compute_full_score(centers_local)
        if score_check > best_score:
            best_score = score_check
            best_centers = centers_local.copy()
    
    # Final refinement: run SA from best solution with smaller steps
    if best_centers is not None and time.time() - start_time < max_time - 10:
        time_left = max_time - (time.time() - start_time)
        if time_left > 10:
            # Create perturbed version for refinement
            refined_init = best_centers + rng.normal(0, 0.01, (N, 2))
            refined_init = np.clip(refined_init, -0.4999, 0.4999)
            
            centers_final, score_final = _sa_run(
                refined_init, rng, start_time, max_time, 100000
            )
            
            score_check = _compute_full_score(centers_final)
            if score_check > best_score:
                best_score = score_check
                best_centers = centers_final.copy()
    
    # Ensure valid output
    if best_centers is None:
        best_centers = initializations[0]
    
    # Final validation and clipping
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
