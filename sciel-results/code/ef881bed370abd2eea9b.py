import numpy as np
import time
import sys

N = 200

def compute_radius(points):
    """
    Computes the valid radius for the given points.
    Points are expected to be in [-0.5, 0.5]^2.
    Returns the minimum of:
    1. Half the minimum distance between any two points.
    2. The minimum distance from any point to the boundary (0.5 - max coord).
    """
    if not np.all(np.isfinite(points)):
        return -1.0
    
    # Check boundaries
    min_coord = points.min(axis=0)
    max_coord = points.max(axis=0)
    
    # Distance to boundaries
    # Points must be inside [-0.5, 0.5]
    # Distance to left wall: x - (-0.5) = x + 0.5
    # Distance to right wall: 0.5 - x
    # Distance to bottom wall: y + 0.5
    # Distance to top wall: 0.5 - y
    
    dist_to_left = points[:, 0] + 0.5
    dist_to_right = 0.5 - points[:, 0]
    dist_to_bottom = points[:, 1] + 0.5
    dist_to_top = 0.5 - points[:, 1]
    
    min_dist_to_boundary = np.min(np.concatenate([dist_to_left, dist_to_right, dist_to_bottom, dist_to_top]))
    
    if min_dist_to_boundary < 0:
        return -1.0

    # Compute pairwise distances
    # Vectorized distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r_pairs = min_dist / 2.0
    r_boundary = min_dist_to_boundary
    
    return min(r_pairs, r_boundary)

def hex_grid_init(n_points, n_restarts):
    """
    Generates diverse hexagonal grid initializations.
    Returns a list of arrays, each of shape (N, 2).
    """
    inits = []
    # Base grid dimensions
    # We want to cover the square with a hex grid.
    # Side length 1.
    # Hex spacing dx, dy = dx * sqrt(3)/2
    # Number of points in a grid of width W and height H:
    # Roughly (W/dx) * (H/dy)
    
    # Let's try different grid densities and offsets
    for i in range(n_restarts):
        # Vary the grid aspect ratio and offset slightly
        # Base grid: 14x15 is 210 points. We need 200.
        # We can generate a slightly larger grid and randomly select 200,
        # or generate a grid that fits exactly.
        # Better: Generate a dense grid, then randomly sample 200 points 
        # with a minimum distance constraint (Poisson disk sampling style) 
        # or just perturb a base grid.
        
        # Let's stick to the strategy: Hex grid init, then perturb.
        # Create a hex grid that covers [-0.5, 0.5]
        
        # Parameters
        # Let's aim for a grid that is slightly denser than needed, 
        # then we can't just take 200 arbitrarily without bias.
        # Instead, let's create a grid that is exactly N points if possible?
        # 200 is not a perfect hex number.
        
        # Alternative: Use a random grid but with a "jitter" that is small.
        # The inspiration mentions "hexagonal grid initialization".
        # Let's create a 15x14 grid (210 points), remove 10 random points.
        
        k_rows = 15
        m_cols = 14
        total = k_rows * m_cols
        
        # Generate base hex grid
        dx = 1.0 / (m_cols + 1)
        dy = dx * np.sqrt(3) / 2.0
        
        points = np.zeros((total, 2))
        idx = 0
        for r in range(k_rows):
            y = (r - (k_rows - 1) / 2.0) * dy
            for c in range(m_cols):
                x = (c - (m_cols - 1) / 2.0) * dx
                if r % 2 != 0:
                    x += dx / 2.0
                points[idx] = [x, y]
                idx += 1
        
        # Scale to fit in [-0.5, 0.5] with some margin
        # Current extent:
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        
        # We want the points to be inside [-0.5, 0.5].
        # Scale factor to fit in [-0.45, 0.45] to allow movement?
        # Or just normalize to [-0.5, 0.5] and let the optimizer handle boundaries.
        # The optimizer will push them in if they hit the wall.
        
        scale = 0.9 / max(max_x - min_x, max_y - min_y)
        points *= scale
        points -= points.mean(axis=0)
        
        # Add small random perturbation to break symmetry
        # The amount of perturbation can vary per restart
        perturbation_scale = 0.01 * (i + 1)
        points += np.random.normal(0, perturbation_scale, size=points.shape)
        
        # Select 200 points
        # If we just take the first 200, we bias towards top-left.
        # Let's shuffle the indices.
        idxs = np.random.permutation(total)
        points = points[idxs[:N]]
        
        # Ensure they are strictly inside for initialization validity?
        # The optimizer will clip or push them.
        # Let's clip to [-0.49, 0.49] to be safe for the first step.
        points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
        points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
        
        inits.append(points)
        
    return inits

def soft_min_gradient(points, r_target, beta):
    """
    Computes the gradient of the soft-min surrogate for the radius.
    We want to maximize the minimum distance.
    
    Surrogate: f(points) = -1/beta * log( sum_{i<j} exp(-beta * d_ij) + sum_i exp(-beta * d_i_wall) )
    This approximates min(d_ij, d_i_wall) as beta -> infinity.
    
    Gradient of f w.r.t. points:
    Let S = sum exp(-beta * d_k)
    f = -1/beta * log(S)
    df/dp_i = -1/beta * (1/S) * dS/dp_i
    dS/dp_i = sum_k exp(-beta * d_k) * (-beta * d d_k/dp_i)
    
    So df/dp_i = (1/S) * sum_k exp(-beta * d_k) * d d_k/dp_i
    
    Note: d_k/dp_i is the unit vector from i to k (for pair i,j) or from i to wall.
    Wait, d d_ij/dp_i = (p_i - p_j) / d_ij.
    
    We want to MAXIMIZE the min distance.
    So we want to move in the direction of -gradient of the "badness" (overlap).
    Or simply, gradient ascent on the soft-min of distances.
    
    Let's define the objective to maximize as M = soft_min(distances).
    grad_M = (1/S) * sum_k w_k * u_k
    where w_k = exp(-beta * d_k) and u_k is the unit vector increasing the distance.
    For pair (i,j): increasing d_ij means moving i away from j. u_ij = (p_i - p_j)/d_ij.
    For wall (i, wall): increasing dist to wall means moving i towards center.
    
    We compute the gradient for each point i.
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Weights for pairs
    # To avoid overflow/underflow, we can shift beta * dist
    # But beta is not too large (e.g., 50-200), dist ~ 0.03. beta*dist ~ 1-6. Safe.
    weights = np.exp(-beta * dist)
    
    # Set diagonal to 0
    np.fill_diagonal(weights, 0)
    
    # Sum of weights for normalization
    S_pairs = np.sum(weights, axis=1)
    
    # Gradient contribution from pairs
    # grad_i = sum_j weights[i,j] * (p_i - p_j) / dist[i,j]
    # We can compute this vectorized.
    # u = diff / dist
    # Handle dist=0 (diagonal is inf, so diff is 0, 0/inf = 0). 
    # But we filled dist diagonal with inf, so diff diagonal is 0.
    # Let's be safe.
    dist_safe = np.where(dist > 1e-10, dist, 1.0)
    u = diff / dist_safe[:, :, np.newaxis]
    
    # Weighted sum of unit vectors
    grad_pairs = np.sum(weights[:, :, np.newaxis] * u, axis=1)
    
    # Normalize by S_pairs
    # If S_pairs is 0, gradient is 0.
    mask = S_pairs > 1e-10
    grad_pairs[mask] /= S_pairs[mask]
    
    # Wall distances
    # d_left = x + 0.5
    # d_right = 0.5 - x
    # d_bottom = y + 0.5
    # d_top = 0.5 - y
    
    x = points[:, 0]
    y = points[:, 1]
    
    d_left = x + 0.5
    d_right = 0.5 - x
    d_bottom = y + 0.5
    d_top = 0.5 - y
    
    # Weights for walls
    w_left = np.exp(-beta * d_left)
    w_right = np.exp(-beta * d_right)
    w_bottom = np.exp(-beta * d_bottom)
    w_top = np.exp(-beta * d_top)
    
    S_walls = w_left + w_right + w_bottom + w_top
    
    # Gradient contribution from walls
    # Moving away from left wall (increasing d_left) means moving +x.
    # u_left = +1 in x.
    # Moving away from right wall (increasing d_right) means moving -x.
    # u_right = -1 in x.
    # Etc.
    
    grad_walls = np.zeros_like(points)
    grad_walls[:, 0] += w_left   # Push right
    grad_walls[:, 0] -= w_right  # Push left
    grad_walls[:, 1] += w_bottom # Push up
    grad_walls[:, 1] -= w_top    # Push down
    
    # Normalize
    mask_w = S_walls > 1e-10
    grad_walls[mask_w] /= S_walls[mask_w, np.newaxis]
    
    # Total gradient
    grad = grad_pairs + grad_walls
    
    # We want to maximize the soft-min.
    # The magnitude of the gradient depends on beta.
    # For gradient ascent, we just use the direction.
    return grad

def optimize_single(points, time_limit):
    """
    Optimizes a single configuration using gradient ascent on the soft-min surrogate.
    """
    start_time = time.time()
    
    # Multi-stage beta schedule
    # Start with low beta (smooth), end with high beta (sharp)
    # beta: 10, 20, 50, 100, 200
    betas = [10, 20, 50, 100, 200]
    
    # Learning rates for each stage
    # As beta increases, the gradient becomes more localized and potentially steeper?
    # Actually, the gradient magnitude is roughly 1 (normalized).
    # But the landscape becomes sharper. Smaller steps are safer.
    lrs = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    # Number of steps per stage
    # We have ~100s total.
    # Let's allocate steps based on time.
    # Or fixed steps and check time.
    
    current_points = points.copy()
    best_points = current_points.copy()
    best_r = compute_radius(current_points)
    
    for stage, beta in enumerate(betas):
        lr = lrs[stage]
        
        # Run optimization until time limit or max steps
        # Let's do 500 steps per stage?
        # 5 stages * 500 steps = 2500 steps.
        # Each step is O(N^2) = 40k ops. Vectorized in NumPy.
        # 2500 * 40k = 100M ops. Should be fast (< 1s per restart if optimized).
        
        for step in range(500):
            if time.time() > time_limit:
                return best_points
            
            # Compute gradient
            grad = soft_min_gradient(current_points, 0, beta) # r_target not used in this formulation directly, beta controls sharpness
            
            # Update
            current_points += lr * grad
            
            # Clip to boundaries
            current_points[:, 0] = np.clip(current_points[:, 0], -0.5, 0.5)
            current_points[:, 1] = np.clip(current_points[:, 1], -0.5, 0.5)
            
            # Check improvement
            # Doing compute_radius every step is expensive (O(N^2)).
            # Let's do it every 10 steps.
            if step % 10 == 0:
                r = compute_radius(current_points)
                if r > best_r:
                    best_r = r
                    best_points = current_points.copy()
                    
    # Final check
    r = compute_radius(current_points)
    if r > best_r:
        best_points = current_points.copy()
        
    return best_points

def construct():
    """
    Constructs the centers of 200 equal circles in a unit square [-0.5, 0.5]^2.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate initializations
    # How many restarts?
    # If each restart takes ~5s, we can do ~20 restarts.
    # Let's estimate time for one restart.
    
    # First, let's do a quick test run to estimate time?
    # No, just allocate based on a safe guess.
    # Let's try 10 restarts first.
    
    n_restarts = 10
    inits = hex_grid_init(N, n_restarts)
    
    best_points = None
    best_r = -1.0
    
    for i in range(n_restarts):
        # Check time
        elapsed = time.time() - start_time
        remaining = deadline - start_time
        
        # Estimate time per restart
        # If we've done i restarts, average time is elapsed / i
        if i > 0:
            avg_time = elapsed / i
        else:
            avg_time = 5.0 # Guess 5s per restart initially
        
        # If remaining time is not enough for another restart, break
        if remaining < avg_time * 1.5:
            break
            
        # Allocate time for this restart
        # Leave a bit of buffer
        restart_deadline = start_time + remaining * 0.9
        if i == 0:
            restart_deadline = start_time + min(10.0, remaining) # Cap first one to 10s?
        else:
            restart_deadline = start_time + remaining * 0.95
            
        # Optimize
        points = optimize_single(inits[i], restart_deadline)
        
        # Evaluate
        r = compute_radius(points)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
            
    # If no points found (shouldn't happen), return random
    if best_points is None:
        best_points = np.random.rand(N, 2) - 0.5
        
    return best_points

# EVOLVE-BLOCK-END
