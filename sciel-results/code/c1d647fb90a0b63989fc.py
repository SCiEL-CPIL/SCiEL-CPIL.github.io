import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0 or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Distance to walls
    # min(0.5 - x, x + 0.5) for each coordinate, then min over all points and coords
    wall_dist = np.min(np.minimum(0.5 - centers, centers + 0.5))
    
    if wall_dist <= 0:
        return 0.0

    # Distance to neighbors
    # Vectorized pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N):
    """Generate a hexagonal grid of points."""
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _directed_relaxation(centers, time_limit, start_time, max_iters=5000):
    """
    Directed geometric relaxation.
    1. Compute all pairwise distances and wall distances.
    2. Find the current bottleneck radius r.
    3. Identify the constraint (pair or wall) that limits r.
    4. Move the involved points to increase that specific distance, 
       while trying not to decrease other distances too much (simple projection).
    """
    centers = centers.copy()
    n = len(centers)
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    # Pre-allocate arrays for speed
    # We will use a simple iterative scheme:
    # Repeat:
    #   1. Compute distances
    #   2. Find min pair dist d_min and min wall dist w_min
    #   3. r = min(d_min/2, w_min)
    #   4. If d_min/2 < w_min: 
    #        Find pair (i,j) with dist d_min.
    #        Move i and j away from each other by a small step.
    #   5. Else:
    #        Find point k with min wall dist.
    #        Move k away from the wall.
    #   6. Project to square.
    
    lr = 0.05 # Step size for movement
    stall_count = 0
    last_best = best_score
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # 1. Compute distances
        # Use cdist for speed
        dist_mat = cdist(centers, centers)
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        
        # Wall distances
        wall_dists = np.minimum(0.5 - centers, centers + 0.5)
        min_wall_dist = np.min(wall_dists)
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_score + 1e-9:
            best_score = current_r
            best_centers = centers.copy()
            stall_count = 0
            # Adaptive learning rate: reduce if improving slowly? 
            # Keep constant for now, maybe decay slightly over time
        else:
            stall_count += 1
            if stall_count > 100:
                lr *= 0.95
                if lr < 1e-5:
                    break
                stall_count = 0

        # 2. Identify bottleneck and move
        if min_pair_dist / 2.0 < min_wall_dist:
            # Bottleneck is pair distance
            # Find the pair
            idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            i_idx, j_idx = idx
            # Ensure i < j to avoid double counting if we were summing, but here we just pick one
            if i_idx > j_idx:
                i_idx, j_idx = j_idx, i_idx
            
            vec = centers[i_idx] - centers[j_idx]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                unit_vec = vec / norm
                # Move i in direction of unit_vec, j in opposite
                # Step size proportional to how much we want to increase distance
                # We want to increase dist by roughly lr * current_r? Or fixed lr?
                # Let's use a fixed fraction of the current gap
                step = lr * 0.1 # Small step
                centers[i_idx] += step * unit_vec
                centers[j_idx] -= step * unit_vec
        else:
            # Bottleneck is wall distance
            # Find the point and wall
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            k_idx, coord_idx = idx
            
            # Determine direction away from wall
            if centers[k_idx, coord_idx] < 0:
                direction = 1.0 # Move towards +
            else:
                direction = -1.0 # Move towards -
            
            step = lr * 0.1
            centers[k_idx, coord_idx] += direction * step
        
        # 3. Project to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optional: Simple local repair if we created a new violation
        # The directed move might push a point into another.
        # A quick pass of force-based repulsion for 50 iters could help?
        # Let's stick to pure directed for now to be distinct, but maybe add a small random jitter to escape local minima
        if i % 10 == 0:
             # Small random perturbation to escape strict local minima of the directed method
             # Only if stalled
             if stall_count > 20:
                 centers += np.random.normal(0, 0.001, centers.shape)
                 centers = np.clip(centers, -0.5, 0.5)

    return best_centers, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    # 1. Generate diverse initializations
    inits = []
    
    # Hex Grid
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    # Square Grid
    square_grid = _generate_square_grid(N)
    inits.append(square_grid)
    
    # Jittered Hex
    if hex_grid is not None:
        np.random.seed(42)
        jittered_hex = hex_grid + np.random.normal(0, 0.01, hex_grid.shape)
        jittered_hex = np.clip(jittered_hex, -0.49, 0.49)
        inits.append(jittered_hex)
        
    # Random Initializations
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)
        
    # Sort initializations by their initial score (best first)
    scored_inits = []
    for init in inits:
        score = _compute_score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Run directed relaxation on each initialization
    for score, init_centers in scored_inits:
        if time.time() - start_time > time_limit:
            break
            
        # Run directed relaxation
        centers, local_score = _directed_relaxation(init_centers, time_limit, start_time)
        
        if local_score > best_score:
            best_score = local_score
            best_centers = centers
            
    # 3. If time remains, do random restarts from the best found so far
    if best_centers is not None and time.time() - start_time < time_limit - 10:
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            np.random.seed(int(time.time() * 1000) % 10000)
            # Perturb best solution
            perturbed = best_centers + np.random.normal(0, 0.02, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            centers, local_score = _directed_relaxation(perturbed, time_limit, start_time)
            
            if local_score > best_score:
                best_score = local_score
                best_centers = centers
                
    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
