"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius for a given set of centers."""
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Minimum distance between pairs
    n = len(centers)
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    return min(min_pair_dist / 2.0, min_boundary)

def _score_fast(centers):
    """Fast score computation using vectorized operations."""
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Vectorized pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_score(centers, beta=20.0):
    """Smooth surrogate: soft-min of (pair_dists/2, boundary_dists).
    Uses log-sum-exp for numerical stability. Returns a smooth approximation
    of the true min-score that is differentiable and guides search."""
    # Boundary distances
    boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Pair distances (upper triangle only to save memory)
    n = len(centers)
    iu = np.triu_indices(n, k=1)
    diff = centers[iu[0]] - centers[iu[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1)) / 2.0
    
    # Combine all constraints into one array
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min via log-sum-exp: soft_min(x) = -logsumexp(-beta*x)/beta
    # For numerical stability, subtract max
    neg_beta_x = -beta * all_dists
    m = np.max(neg_beta_x)
    log_sum_exp = m + np.log(np.sum(np.exp(neg_beta_x - m)))
    soft_min_val = -log_sum_exp / beta
    
    return soft_min_val

def _polish_coordinate_descent(centers, start_time, deadline, max_sweeps=500):
    """High-precision polish: coordinate descent on individual coordinates.
    For each coordinate, find the best value that maximizes the min-score."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    coord_step = 1e-4
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        for i in range(N):
            for dim in range(2):
                # Try small perturbations in this coordinate
                best_val = best_centers[i, dim]
                best_s = best_score
                
                # Scan around current value
                for delta in np.linspace(-coord_step, coord_step, 21):
                    if delta == 0:
                        continue
                    trial = best_centers.copy()
                    trial[i, dim] = np.clip(best_centers[i, dim] + delta, -0.5, 0.5)
                    s = _score_fast(trial)
                    if s > best_s + 1e-15:
                        best_s = s
                        best_val = trial[i, dim]
                
                if best_val != best_centers[i, dim]:
                    best_centers[i, dim] = best_val
                    improved = True
        
        if best_score < best_s - 1e-15:
            best_score = best_s
        
        # Shrink step as we converge
        if sweep % 20 == 19:
            coord_step *= 0.7
        
        if not improved and coord_step < 1e-7:
            break
    
    return best_centers, best_score

def _optimize_from_start(centers, start_time, deadline, max_iter=3000):
    """Optimize from a starting configuration using smooth surrogate-guided search."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    
    # Adaptive step size
    step_size = 0.015
    patience = 0
    max_patience = 80
    
    for it in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        # Generate perturbation
        if patience > max_patience:
            # Reset to best and increase step size
            centers = best_centers.copy()
            step_size *= 1.3
            patience = 0
        
        # Random perturbation with occasional large jumps
        if np.random.random() < 0.05:
            # Large jump to escape local optima
            noise = np.random.randn(N, 2) * step_size * 3.0
        else:
            noise = np.random.randn(N, 2) * step_size
        
        new_centers = centers + noise
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Use soft-min as primary guide, exact score for acceptance
        new_soft = _soft_min_score(new_centers, beta=15.0)
        cur_soft = _soft_min_score(centers, beta=15.0)
        
        # Accept if soft score improves OR exact score improves
        new_score = _score_fast(new_centers)
        
        if new_score > best_score or (new_soft > cur_soft and new_score > best_score - 1e-6):
            best_score = max(best_score, new_score)
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            patience = 0
            step_size *= 0.97  # Slightly decrease step size on improvement
        else:
            patience += 1
            step_size *= 0.93  # Decrease step size on failure
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Generate multiple initial configurations and optimize each
    
    # Method 1: Hexagonal grid variations
    for n_cols in range(5, 12):
        for n_rows in range(5, 12):
            if n_cols * n_rows < N:
                continue
            
            dx = 2.0 / n_cols
            dy = dx * np.sqrt(3) / 2.0
            
            total_w = (n_cols - 1) * dx + dx/2
            total_h = (n_rows - 1) * dy
            
            offset_x = -total_w / 2.0
            offset_y = -total_h / 2.0
            
            centers = []
            for i in range(n_rows):
                for j in range(n_cols):
                    x = offset_x + j * dx + (i % 2) * dx / 2.0
                    y = offset_y + i * dy
                    centers.append([x, y])
            
            centers = np.array(centers[:N])
            
            if len(centers) < N:
                continue
            
            if np.any(np.abs(centers) > 0.5):
                continue
            
            s = _score_fast(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
            
            if time.time() - start_time > deadline:
                break
    
    # Method 2: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    s = _score_fast(grid)
    if s > best_score:
        best_score = s
        best_centers = grid.copy()
    
    # Method 3: Random initialization with multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Random points
        centers = np.random.uniform(-0.45, 0.45, (N, 2))
        s = _score_fast(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Optimize from the best initial configuration
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5:
        # Phase 1: Main optimization with smooth surrogate
        phase1_deadline = start_time + min(remaining_time - 2.0, 90.0)
        optimized_centers, opt_score = _optimize_from_start(
            best_centers, start_time, phase1_deadline, max_iter=5000
        )
        if opt_score > best_score:
            best_score = opt_score
            best_centers = optimized_centers
        
        # Phase 2: High-precision coordinate descent polish
        elapsed = time.time() - start_time
        if elapsed < deadline - 3.0:
            polished_centers, polish_score = _polish_coordinate_descent(
                best_centers, start_time, deadline - 2.0, max_sweeps=200
            )
            if polish_score > best_score:
                best_score = polish_score
                best_centers = polished_centers
    
    return best_centers
# EVOLVE-BLOCK-END
