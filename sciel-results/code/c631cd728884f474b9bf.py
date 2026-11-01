"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def _calc_radius(pts):
    """Compute the maximum valid radius for given point positions."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_dense_hex(n):
    """
    Generate a dense hexagonal lattice and select n points optimally.
    Uses multiple spacing values and selects the best subset.
    """
    best_points = None
    best_score = -1.0
    
    # Try a range of spacings. 
    # Optimal spacing for N points in unit square is roughly 1/sqrt(N) ~ 0.07
    # Hexagonal packing density is higher, so spacing can be slightly larger than grid.
    spacings = np.linspace(0.065, 0.085, 25)
    
    for d in spacings:
        row_h = d * np.sqrt(3.0) / 2.0
        col_w = d
        
        # Generate all lattice points in [-0.5-d, 0.5+d]^2
        # We need to be careful with arange boundaries
        y_start = -0.5 - row_h
        y_end = 0.5 + row_h
        y_vals = np.arange(y_start, y_end, row_h)
        
        all_pts = []
        
        for j, y in enumerate(y_vals):
            # Only consider points that might fit in y-range
            if y < -0.5 or y > 0.5:
                continue
            
            offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
            x_start = -0.5 - col_w + offset
            x_end = 0.5 + col_w + offset
            x_vals = np.arange(x_start, x_end, col_w)
            
            for x in x_vals:
                if -0.5 <= x <= 0.5:
                    all_pts.append([x, y])
        
        if len(all_pts) < n:
            continue
        
        all_pts = np.array(all_pts)
        
        # Score each candidate point by how "central" it is
        # Points closer to boundary are more likely to limit radius
        dist_to_bnd = np.minimum(
            0.5 - np.abs(all_pts[:, 0]),
            0.5 - np.abs(all_pts[:, 1])
        )
        
        # Sort by distance to boundary (descending) - take most central points
        idx = np.argsort(dist_to_bnd)[::-1]
        selected = all_pts[idx[:n]]
        
        score = _calc_radius(selected)
        if score > best_score:
            best_score = score
            best_points = selected.copy()
    
    if best_points is None:
        # Fallback: regular grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    return best_points

def _repulsion_optimize(points, max_iter=8000, power=4):
    """
    Optimize point positions to maximize minimum distance using
    gradient ascent on a repulsive potential + boundary penalties.
    
    Objective: Minimize E = sum_{i<j} (1/d_ij^power) + sum_i (1/b_i^power)
    where b_i is distance to nearest boundary.
    Minimizing this pushes points apart and away from boundaries.
    """
    n = len(points)
    points = points.copy()
    
    # Adaptive learning rate
    lr = 0.005
    min_lr = 1e-6
    decay = 0.995
    
    # Track best radius
    best_r = _calc_radius(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        # Compute pairwise differences and distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Gradient of sum(1/d^power)
        # d/dx_i (1/d^p) = -p * (x_i - x_j) / d^(p+2)
        # We want to minimize E, so we move in direction -grad(E).
        # grad(E)_i = sum_{j!=i} -p * (x_i - x_j) / d^(p+2)
        # So update: x_i += lr * p * sum_{j!=i} (x_i - x_j) / d^(p+2)
        # This is repulsion.
        
        # Vectorized calculation
        # weights = p / dists^(p+2)
        # grad_i = sum_j weights_ij * (x_i - x_j)
        # Note: (x_i - x_j) is row i of diff.
        
        # Avoid division by zero issues, though dists diag is inf
        weights = power / (dists ** (power + 2))
        np.fill_diagonal(weights, 0.0)
        
        # grad_pair[i] = sum_j weights[i,j] * (points[i] - points[j])
        # = sum_j weights[i,j] * points[i] - sum_j weights[i,j] * points[j]
        # = points[i] * sum_j weights[i,j] - (weights @ points)[i]
        
        row_sums = np.sum(weights, axis=1)
        weighted_sum = weights @ points
        grad_pair = row_sums[:, np.newaxis] * points - weighted_sum
        
        # Boundary Gradient
        # b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # We want to minimize 1/b^power.
        # If x-boundary is active (0.5 - |x| < 0.5 - |y|):
        #   b = 0.5 - |x|
        #   db/dx = -sign(x)
        #   d(1/b^p)/dx = -p * b^(-p-1) * (-sign(x)) = p * sign(x) / b^(p+1)
        #   Minimize E => move in -grad direction: x -= lr * p * sign(x) / b^(p+1)
        #   This pushes x towards 0 (center).
        
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        # Determine active boundary
        active_x = bx <= by
        
        # Calculate gradient components
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        
        # For x-active points
        mask_x = active_x
        if np.any(mask_x):
            b_x = bx[mask_x]
            # Avoid div by 0
            b_x = np.maximum(b_x, 1e-6)
            # Gradient of penalty is positive (if x>0) or negative (if x<0)
            # We subtract gradient to minimize.
            # So we want to push x towards 0.
            # Force = p * sign(x) / b^(p+1)
            force_x = power * np.sign(points[mask_x, 0]) / (b_x ** (power + 1))
            grad_x[mask_x] = force_x
            
        # For y-active points
        mask_y = ~active_x
        if np.any(mask_y):
            b_y = by[mask_y]
            b_y = np.maximum(b_y, 1e-6)
            force_y = power * np.sign(points[mask_y, 1]) / (b_y ** (power + 1))
            grad_y[mask_y] = force_y
        
        total_grad = np.column_stack([grad_x, grad_y])
        
        # Update
        points = points - lr * total_grad
        
        # Project to square (hard constraint)
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Check radius
        current_r = _calc_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Adaptive LR: decrease slowly
        lr *= decay
        if lr < min_lr:
            lr = min_lr
            
    return best_points

def construct():
    # Step 1: Generate initial hexagonal lattice
    points = _generate_dense_hex(N)
    
    # Step 2: Optimize using repulsion-based gradient descent
    # Run for a long time to converge
    points = _repulsion_optimize(points, max_iter=10000, power=4)
    
    # Step 3: Final polish with higher power to sharpen distances
    points = _repulsion_optimize(points, max_iter=5000, power=8)
    
    return points
# EVOLVE-BLOCK-END
