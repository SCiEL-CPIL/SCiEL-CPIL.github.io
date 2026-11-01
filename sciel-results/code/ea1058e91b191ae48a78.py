import numpy as np
from scipy.spatial.distance import cdist

N = 50

def construct():
    """
    Constructs the centers of 50 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum distance between circles and to the boundary.
    Uses sub-gradient ascent on the minimum distance function.
    """
    
    # 1. Initialization
    # We use a hexagonal grid initialization similar to the previous version 
    # but we will rely heavily on the optimization step.
    # We'll stick to a simple, robust hex init to ensure we start in a good basin.
    
    def create_hex_initialization():
        # A simple hex grid that fits 50 points reasonably well.
        # 7 rows: 8,7,8,7,8,7,5 is a bit uneven. 
        # Let's try to generate a grid and center it.
        # We want to maximize the initial min distance.
        
        best_points = None
        best_score = -1
        
        # Try a few common configurations
        configs = [
            [8, 7, 8, 7, 8, 7, 5],
            [7, 8, 7, 8, 7, 8, 5],
            [8, 8, 7, 8, 7, 8, 4],
            [7, 7, 8, 8, 8, 7, 5],
        ]
        
        for config in configs:
            if sum(config) != N:
                continue
            
            points = []
            # Generate unit hex grid
            for i, count in enumerate(config):
                y = -(len(config) - 1) / 2.0 + i
                x_start = -(count - 1) / 2.0
                if i % 2 == 1:
                    x_start += 0.5
                for j in range(count):
                    x = x_start + j
                    points.append([x, y])
            
            points = np.array(points)
            # Center
            points -= points.mean(axis=0)
            
            # Scale to fit
            # We need to scale such that min_dist is maximized.
            # For a hex grid, the limiting factor is usually the boundary or the densest part.
            # Let's just scale to fit the bounding box with some margin.
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            width = max_x - min_x
            height = max_y - min_y
            
            # Scale factor s. Distance between neighbors becomes s.
            # Boundary distance is 0.5 - s*max(|x|,|y|).
            # We want to maximize min(s, 0.5 - s*max_norm).
            # Let's just scale to fit inside [-0.45, 0.45] roughly, then let optimizer fix.
            scale = 0.9 / max(width, height)
            points *= scale
            
            # Calculate score
            # Use cdist for speed
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            min_pair = dists.min()
            
            # Boundary
            max_norm = np.max(np.abs(points), axis=1)
            min_bound = 0.5 - max_norm.max()
            
            score = min(min_pair, min_bound)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        return best_points

    points = create_hex_initialization()
    
    # 2. Sub-gradient Ascent Optimization
    # We maximize r = min(d_ij, d_i_wall)
    
    def get_min_dist_and_active(pts):
        # Returns min_dist, list of active pairs (i,j), list of active walls (i, axis, sign)
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # Find min pair distance
        min_pair_val = dists.min()
        active_pairs = []
        # Tolerance to include near-minimal pairs to get a smoother gradient
        tol = 1e-6 
        # To be more robust, let's take the top K smallest distances or those within a small epsilon
        # For sub-gradient, we usually take the exact minimum, but using a set of near-minimums
        # can help escape saddle points. Let's use a small window.
        # However, strictly speaking, the sub-gradient of min is the average of gradients of active mins.
        # Let's just use the exact minimums for now, plus maybe a tiny epsilon.
        
        # Get indices where dist is close to min
        # Using a small epsilon to handle numerical noise
        eps = 1e-8
        active_mask = dists <= (min_pair_val + eps)
        # Get pairs
        i_idx, j_idx = np.where(active_mask)
        # Ensure i < j to avoid duplicates if we were summing, but for gradient we need both
        # Actually, for gradient on i, we need vector from j to i.
        # For gradient on j, we need vector from i to j.
        
        # Boundary distances
        # d_i_wall = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # Let's compute distances to each of the 4 walls
        d_left = 0.5 + pts[:, 0]   # dist to x=-0.5
        d_right = 0.5 - pts[:, 0]  # dist to x=0.5
        d_down = 0.5 + pts[:, 1]   # dist to y=-0.5
        d_up = 0.5 - pts[:, 1]     # dist to y=0.5
        
        min_bound_val = np.min(np.min([d_left.min(), d_right.min(), d_down.min(), d_up.min()]))
        
        min_dist = min(min_pair_val, min_bound_val)
        
        # Determine which constraint is active (pair or wall)
        # If min_dist is from pair, we use pair gradients.
        # If from wall, we use wall gradients.
        # It's possible both are equal. We should combine them.
        
        grads = np.zeros_like(pts)
        
        if min_dist == min_pair_val:
            # Sum gradients for all active pairs
            for i, j in zip(i_idx, j_idx):
                # Vector from j to i
                v = pts[i] - pts[j]
                d = np.linalg.norm(v)
                if d > 1e-12:
                    v_norm = v / d
                    grads[i] += v_norm
                    grads[j] -= v_norm
                    # Note: We are maximizing distance. 
                    # Gradient of d_ij wrt c_i is (c_i - c_j)/d_ij.
                    # So moving in direction of grad increases d_ij.
        
        if min_dist == min_bound_val:
            # Find active walls
            # For each point, check which wall is at min_bound_val
            for i in range(N):
                # Check left
                if abs(d_left[i] - min_dist) < 1e-8:
                    grads[i, 0] += 1.0 # Move right
                # Check right
                if abs(d_right[i] - min_dist) < 1e-8:
                    grads[i, 0] -= 1.0 # Move left
                # Check down
                if abs(d_down[i] - min_dist) < 1e-8:
                    grads[i, 1] += 1.0 # Move up
                # Check up
                if abs(d_up[i] - min_dist) < 1e-8:
                    grads[i, 1] -= 1.0 # Move down
                    
        return min_dist, grads

    current_min_dist, grads = get_min_dist_and_active(points)
    
    # Optimization loop
    # We use a decreasing step size.
    # We also add a small random jitter to escape local minima if stuck.
    
    max_iter = 5000
    step_size = 0.05
    decay = 0.999
    stuck_count = 0
    best_min_dist = current_min_dist
    best_points = points.copy()
    
    for iter in range(max_iter):
        # Compute gradient
        current_min_dist, grads = get_min_dist_and_active(points)
        
        if current_min_dist > best_min_dist:
            best_min_dist = current_min_dist
            best_points = points.copy()
            stuck_count = 0
        else:
            stuck_count += 1
        
        # If stuck for a while, increase jitter or restart step size?
        # For now, just continue.
        
        # Normalize gradients to prevent large steps from many active constraints
        # Each point's gradient is a vector. We normalize it to unit length if non-zero.
        # This makes the step size consistent.
        norm_grads = np.linalg.norm(grads, axis=1, keepdims=True)
        norm_grads[norm_grads == 0] = 1.0
        grads = grads / norm_grads
        
        # Add small random noise to handle symmetry breaking / flat spots
        noise = np.random.normal(0, 0.01, size=points.shape)
        
        # Update
        points += step_size * (grads + noise)
        
        # Project back into square [-0.5, 0.5]
        # We must ensure points stay inside.
        points = np.clip(points, -0.5, 0.5)
        
        # Decay step size
        step_size *= decay
        
        # Early stopping if step size is too small
        if step_size < 1e-6:
            break
            
    # Final check: ensure we return the best found
    # The "best" found might be slightly better than the current state if we clipped and broke constraints?
    # Actually, clipping might reduce the min distance if a point was pushed out.
    # So we should check the score of best_points vs points at the end.
    
    # Recalculate score for best_points to be sure
    # We just return best_points which was stored when it was the best.
    
    return best_points

# EVOLVE-BLOCK-END
