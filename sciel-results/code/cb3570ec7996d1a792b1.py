import numpy as np
import time

N = 300

def construct():
    # 1. Initialize with hexagonal lattice
    # Approximate spacing for N=300 in unit circle
    # Area of unit circle = pi. Area of hex cell ~ sqrt(3)/2 * s^2.
    # N * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate hex lattice points
    points = []
    # Range to cover unit circle
    R = 1.5
    k = int(R / s_init) + 2
    for j in range(-k, k + 1):
        y = j * s_init * np.sqrt(3) / 2
        x_offset = (j % 2) * s_init / 2
        for i in range(-k, k + 1):
            x = i * s_init + x_offset
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Fallback: random points
        points = np.random.randn(N, 2) * 0.5
    else:
        # Select N points closest to center
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
        
    # Ensure finite and valid
    points = np.nan_to_num(points)
    
    x = points.copy()
    
    def calc_score(x):
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)
    
    best_x = x.copy()
    best_r = calc_score(best_x)
    
    start_time = time.time()
    max_time = 110.0 # Safety margin for 120s limit
    
    # Optimization parameters
    alpha = 0.05 # Initial step size
    k_closest = 50 # Number of closest pairs/points to consider
    
    # We will run the optimization in phases or with a resetting alpha
    while time.time() - start_time < max_time:
        # 1. Compute distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        bound_dists = 1.0 - np.linalg.norm(x, axis=1)
        
        # Determine current r
        min_pair_val = np.min(dists) / 2.0
        min_bound_val = np.min(bound_dists)
        current_r = min(min_pair_val, min_bound_val)
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_x = x.copy()
            
        # Compute Gradient
        grad = np.zeros_like(x)
        
        # Threshold for "active" constraints
        # We want to push apart pairs that are close to the minimum
        # and pull in points close to boundary.
        tol = 0.05 * current_r + 1e-4
        
        # Pair forces
        # Get k smallest pair distances
        flat_dists = dists.flatten()
        # We need indices in 2D
        # To be efficient, we can just take the top k from the flattened array
        # But we need to be careful with duplicates (i,j) and (j,i)
        # Using argsort on the whole N^2 array is O(N^2 log N). For N=300, 90000 elements, this is fast.
        flat_indices = np.argsort(flat_dists)
        
        # Process top k_closest * 2 indices (to account for symmetry roughly)
        # Actually, since dists is symmetric, if (i,j) is small, (j,i) is also small.
        # Taking top 2*k ensures we cover the unique pairs well.
        for idx in flat_indices[:k_closest * 2]:
            i, j = np.unravel_index(idx, dists.shape)
            if i == j: continue
            d = dists[i, j]
            if d < current_r * 2 + tol * 2:
                v = x[i] - x[j]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-8:
                    v = v / norm_v
                    grad[i] += v
                    grad[j] -= v
        
        # Boundary forces
        bound_indices = np.argsort(bound_dists)
        for idx in bound_indices[:k_closest]:
            i = idx
            d_b = bound_dists[i]
            if d_b < current_r + tol:
                v = -x[i]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-8:
                    v = v / norm_v
                    grad[i] += v
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
            
        # Update
        x += alpha * grad
        
        # Project back to unit circle if any point goes out
        norms = np.linalg.norm(x, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            x[out_mask] /= norms[out_mask, np.newaxis]
            
        # Annealing schedule: decay alpha, but reset if it gets too small
        alpha *= 0.995
        if alpha < 0.005:
            alpha = 0.02 # Reset to explore
            # Add small random perturbation to escape local minima
            x += np.random.randn(N, 2) * 0.001
            # Re-project
            norms = np.linalg.norm(x, axis=1)
            out_mask = norms > 1.0
            if np.any(out_mask):
                x[out_mask] /= norms[out_mask, np.newaxis]

    return best_x
