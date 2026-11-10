"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200
TIME_LIMIT = 140.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance
    # Use scipy if available for speed, else fallback
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
    except ImportError:
        # Fallback: compute all pairwise distances
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centres[i] - centres[j])
                if d < min_dist:
                    min_dist = d
    return min(min_dist / 2.0, min_wall)


def _grid_packing(m, n):
    """Create a grid packing with m columns and n rows, total >= N."""
    # We want to place N points in an m x n grid within [-0.5, 0.5]^2
    # But we only use N of them. Let's think differently.
    # Actually, for a grid of m x n cells, if m*n >= N, we can place N points.
    # But the spacing depends on how we arrange them.
    # Better: create a grid that exactly fits N points if possible, or close.
    
    # Find factorization of N into m x n where m*n >= N and m*n is close to N
    # Actually, we just need m*n >= N. Let's try various m.
    best = None
    best_r = -1
    for m in range(1, N + 1):
        n = int(np.ceil(N / m))
        # Grid spacing
        dx = 1.0 / m
        dy = 1.0 / n
        # Radius is min(dx/2, dy/2)
        r = min(dx, dy) / 2.0
        if r > best_r:
            best_r = r
            best = (m, n)
    m, n = best
    # Create grid centres
    x_coords = (np.arange(m) + 0.5) * (1.0 / m) - 0.5
    y_coords = (np.arange(n) + 0.5) * (1.0 / n) - 0.5
    grid = np.array([(x, y) for y in y_coords for x in x_coords])
    # Take first N points
    return grid[:N]


def _hexagonal_packing():
    """Create a hexagonal packing pattern."""
    # Hexagonal packing: rows offset by half a spacing
    # For N points, find optimal number of rows/columns
    best_centres = None
    best_r = -1
    
    for n_rows in range(1, N + 1):
        n_cols = int(np.ceil(N / n_rows))
        # Hexagonal spacing
        # Horizontal spacing: s
        # Vertical spacing: s * sqrt(3)/2
        # In a unit square, we need to fit n_cols points horizontally and n_rows vertically
        # Width used: (n_cols - 1) * s + s = n_cols * s (if we center)
        # Actually, let's think of it as:
        # Each row has n_cols points (or n_cols-1 for odd rows)
        # Let's use a simpler approach: place points in a hex grid and take the best N
        
        # Spacing s such that n_cols * s <= 1 and n_rows * s * sqrt(3)/2 <= 1
        # s = min(1/n_cols, 2/(n_rows * sqrt(3)))
        s = min(1.0 / n_cols, 2.0 / (n_rows * np.sqrt(3)))
        r = s / 2.0
        if r > best_r:
            # Create the grid
            centres = []
            for row in range(n_rows):
                y = -0.5 + (row + 0.5) * s * np.sqrt(3) / 2
                # Offset for odd rows
                offset = 0 if row % 2 == 0 else s / 2
                for col in range(n_cols):
                    x = -0.5 + (col + 0.5) * s + offset
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        centres.append((x, y))
            centres = np.array(centres)
            if len(centres) >= N:
                centres = centres[:N]
                # Recompute actual radius
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    return _grid_packing(int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N))))


def _local_optimization(centres, max_iters=200, step_size=0.02):
    """Simple gradient-free local optimization using random perturbations."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10) # Leave 10s buffer for other parts
    
    from scipy.spatial.distance import cdist
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate constraints to find the most constrained point
        # This is O(N^2) but we do it once per iteration
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)
        
        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        idx = np.argmin(constraints)
        
        # Try a few random perturbations on the most constrained point
        improved = False
        for _ in range(5):
            if time.time() > deadline:
                break
            delta = np.random.normal(0, step_size, size=2)
            new_centres = best_centres.copy()
            new_centres[idx] += delta
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            # Fast check: only compute radius if the move is potentially valid
            # We can optimize _compute_radius, but for now just call it.
            # To speed up, we can check if the new point is closer to neighbors than before
            # But _compute_radius is the ground truth. 
            # Optimization: Only compute full radius if local neighborhood looks better?
            # For simplicity and correctness, we call _compute_radius.
            # However, to save time, we can skip the full N^2 calc if we know it won't improve?
            # No, the global min might be elsewhere. But if we only move one point, 
            # the new global min is min(old_global_min_except_involving_idx, new_local_min).
            # We don't have old_global_min_except_involving_idx easily.
            # So we just call _compute_radius. It's O(N^2) ~ 40k ops. 200 iters * 5 tries = 1000 calls.
            # 1000 * 40k = 40M ops. Should be fine in < 1s with NumPy.
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centres = new_centres
                improved = True
                break
                
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Start with a good initial configuration
    # Try hexagonal packing
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # Local optimization
    if time.time() - start_time < TIME_LIMIT - 15:
        centres, current_r = _local_optimization(centres)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
# EVOLVE-BLOCK-END
