"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _score_fast(centers):
    """Fast score computation using cKDTree for O(N log N) nearest-neighbor queries."""
    if len(centers) == 0:
        return 0.0
    # Boundary distances
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Nearest neighbor distances using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)  # k=2 to get self + nearest neighbor
    min_pair_dist = np.min(dists[:, 1])  # distance to nearest other point
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_score(centers, beta=20.0):
    """Smooth surrogate using soft-min of all constraint distances."""
    # Boundary distances
    boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Pair distances via cKDTree (only need nearest neighbors for the min)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min via log-sum-exp
    neg_beta_x = -beta * all_dists
    m = np.max(neg_beta_x)
    log_sum_exp = m + np.log(np.sum(np.exp(neg_beta_x - m)))
    soft_min_val = -log_sum_exp / beta
    
    return soft_min_val

def _polish_coordinate_descent(centers, start_time, deadline, max_sweeps=500):
    """High-precision polish: coordinate descent optimizing the min-score directly."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    coord_step = 5e-4  # Start with a small step for polish
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        for i in range(N):
            for dim in range(2):
                best_val = best_centers[i, dim]
                best_s = best_score
                
                # Binary search / Golden section search for optimal coordinate value
                # We search in a small window around the current value
                lo = max(-0.5, best_val - coord_step)
                hi = min(0.5, best_val + coord_step)
                
                # Try several candidate values
                candidates = np.linspace(lo, hi, 15)
                for trial_val in candidates:
                    trial = best_centers.copy()
                    trial[i, dim] = trial_val
                    s = _score_fast(trial)
                    if s > best_s + 1e-14:
                        best_s = s
                        best_val = trial_val
                
                if abs(best_val - best_centers[i, dim]) > 1e-12:
                    best_centers[i, dim] = best_val
                    improved = True
        
        if best_score < best_s - 1e-14:
            best_score = best_s
            
        # Shrink step as we converge
        if sweep % 10 == 9:
            coord_step *= 0.8
        
        if not improved and coord_step < 1e-9:
            break
    
    return best_centers, best_score

def _optimize_from_start(centers, start_time, deadline, max_iter=2000):
    """Optimize using smooth surrogate-guided search with adaptive step size."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    
    step_size = 0.005  # Start smaller for local refinement
    patience = 0
    max_patience = 50
    best_soft = _soft_min_score(centers, beta=15.0)
    
    for it in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        if patience > max_patience:
            # Reset to best and increase step size to escape local minima
            centers = best_centers.copy()
            step_size *= 1.5
            patience = 0
        
        # Generate perturbation
        # Mix of small and medium steps
        if np.random.random() < 0.1:
            noise = np.random.randn(N, 2) * step_size * 3.0
        else:
            noise = np.random.randn(N, 2) * step_size
        
        new_centers = centers + noise
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_soft = _soft_min_score(new_centers, beta=15.0)
        new_score = _score_fast(new_centers)
        
        # Accept if exact score improves, or soft score improves significantly
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            best_soft = new_soft
            patience = 0
            step_size *= 0.97
        elif new_soft > best_soft + 0.0005 and new_score > best_score - 1e-6:
            # Accept if soft score improves and we're not too far from best
            centers = new_centers.copy()
            best_soft = new_soft
            patience = 0
            step_size *= 0.98
        else:
            patience += 1
            step_size *= 0.95
    
    return best_centers, best_score

def _generate_hex_pattern(row_lengths, global_scale, offset_x, offset_y):
    """
    Generate centers for a hexagonal pattern defined by row lengths.
    row_lengths: list of ints, e.g., [7, 8, 7, 8, 7, 7, 6]
    global_scale: factor for the base hex spacing
    offset_x, offset_y: global translation
    """
    # Base hex spacing for unit distance between centers
    # If we want radius r, distance is 2r.
    # Here we generate unit-spaced hex lattice and then scale/translate.
    # Actually, it's easier to define the spacing directly.
    
    # Let dx be the horizontal spacing.
    # In a hex lattice, dy = dx * sqrt(3) / 2.
    # The "global_scale" will be applied to dx.
    dx = 1.0 * global_scale
    dy = dx * np.sqrt(3) / 2.0
    
    centers = []
    current_y = offset_y
    
    for i, length in enumerate(row_lengths):
        # Center this row horizontally
        # Row width is (length - 1) * dx
        # Start x should be such that the row is centered around 0 (before global offset)
        # But we need to align with the hex lattice structure.
        # Standard hex lattice:
        # Even rows: x = -dx/2, 0, dx/2, ... (if length is odd) or -dx, 0, ... (if even? no)
        # Let's define the x positions relative to the center of the row.
        
        # For a row of length L, the x offsets from the row center are:
        # If L is odd: (-(L-1)/2 * dx, ..., 0, ..., (L-1)/2 * dx)
        # If L is even: (-(L/2 - 0.5) * dx, ..., (L/2 - 0.5) * dx)
        
        # Actually, to maintain hex connectivity, the rows should be staggered.
        # Let's assume the "center" of the hex lattice is at (0,0).
        # We will generate the points in a local coordinate system and then apply global offset.
        
        # Let's just generate points with x in [-0.5, 0.5] range roughly.
        # It's easier to generate the grid, compute its bounding box, and then center it.
        
        row_centers = []
        for j in range(length):
            # x position in this row
            # We want the rows to be staggered by dx/2.
            # Let's define the x-coordinate of the first point in row i.
            # If we center the entire assembly, we can just generate relative positions.
            
            # Relative x for j-th point in a row of length L:
            # x_rel = (j - (L-1)/2.0) * dx
            x_rel = (j - (length - 1) / 2.0) * dx
            
            # Stagger: even rows shift by dx/2?
            # In a standard hex packing:
            # Row 0: x = 0, dx, 2dx...
            # Row 1: x = dx/2, 3dx/2...
            # Let's use: x_shift = (i % 2) * dx / 2.0
            
            x = x_rel + (i % 2) * dx / 2.0
            y = i * dy
            
            row_centers.append([x, y])
        
        centers.extend(row_centers)
        current_y += dy
        
    centers = np.array(centers)
    
    # Now center the entire assembly
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    centers[:, 0] -= center_x
    centers[:, 1] -= center_y
    
    # Apply global scale and offset
    # Note: The 'global_scale' was already applied to dx/dy in the generation.
    # So we just need to apply the offset.
    centers[:, 0] += offset_x
    centers[:, 1] += offset_y
    
    return centers

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = -np.inf
    
    initial_configs = []
    
    # Define several hexagonal row length patterns that sum to 50
    # Common patterns for 50 circles in a square (approx 7x8 or 8x7)
    patterns = [
        [8, 7, 8, 7, 8, 7, 5], # Sum 50
        [7, 8, 7, 8, 7, 8, 5], # Sum 50
        [8, 8, 7, 7, 7, 8, 5], # Sum 50
        [7, 7, 8, 8, 7, 7, 6], # Sum 50
        [6, 7, 8, 8, 8, 7, 6], # Sum 50
        [8, 7, 8, 7, 8, 7, 5], # Duplicate, but maybe different offsets
        [7, 8, 7, 8, 7, 8, 5],
        [5, 8, 7, 8, 7, 8, 7], # Sum 50
    ]
    
    # Try different global scales and offsets for each pattern
    # We want to fit the pattern in the square [-0.5, 0.5]
    # The "natural" size of the pattern depends on the scale.
    # We will try scales from 0.8 to 1.2 (relative to a baseline)
    
    # Baseline: if we pack 50 circles in a square, r ~ 0.07
    # Distance ~ 0.14
    # Width ~ 7 * 0.14 = 0.98
    # So scale 1.0 should be close.
    
    for pattern in patterns:
        if sum(pattern) != N:
            continue
            
        # Try a range of scales
        for scale in np.linspace(0.9, 1.1, 11):
            # Try a range of offsets (small shifts)
            for ox in np.linspace(-0.05, 0.05, 5):
                for oy in np.linspace(-0.05, 0.05, 5):
                    centers = _generate_hex_pattern(pattern, scale, ox, oy)
                    
                    # Check if all centers are within the square
                    if np.any(np.abs(centers) > 0.5):
                        # If they are outside, we can still try, but the score will be 0 or negative
                        # Actually, the score function handles this by taking min with boundary dist.
                        # If a point is outside, boundary dist is negative.
                        # We should probably skip these or penalize heavily.
                        # For initialization, let's only keep valid ones.
                        continue
                    
                    s = _score_fast(centers)
                    if s > 0.05: # Filter out very bad configs
                        initial_configs.append((s, centers.copy()))
        
        # Also try random offsets for the best scale?
        # Let's just take the top few from the grid search above.
        
    # If the grid search didn't yield enough, add some random ones
    if len(initial_configs) < 5:
        for r in range(10):
            centers = np.random.uniform(-0.48, 0.48, (N, 2))
            s = _score_fast(centers)
            initial_configs.append((s, centers.copy()))
    
    # Sort by score and keep top configurations
    initial_configs.sort(key=lambda x: x[0], reverse=True)
    top_configs = [c for _, c in initial_configs[:8]]
    
    if not top_configs:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 1: Optimize from each top configuration
    for config in top_configs:
        if time.time() - start_time > deadline - 15:
            break
        
        elapsed = time.time() - start_time
        remaining = deadline - 15 - elapsed
        if remaining < 5:
            break
        
        phase_deadline = start_time + min(remaining, 12.0)
        optimized_centers, opt_score = _optimize_from_start(
            config, start_time, phase_deadline, max_iter=1500
        )
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = optimized_centers.copy()
    
    # Phase 2: High-precision coordinate descent polish
    if best_centers is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            polish_deadline = deadline - 3.0
            polished_centers, polish_score = _polish_coordinate_descent(
                best_centers, start_time, polish_deadline, max_sweeps=300
            )
            if polish_score > best_score:
                best_score = polish_score
                best_centers = polished_centers
    
    if best_centers is None:
        # Final fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
