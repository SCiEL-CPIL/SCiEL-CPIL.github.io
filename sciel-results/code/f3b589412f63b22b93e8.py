"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def _generate_hex_grid(n):
    """
    Generate a hexagonal lattice and select n points.
    Strategy: Generate a grid with spacing close to optimal, 
    then select the n points that are furthest from the boundary 
    (to maximize the minimum distance to the wall).
    """
    # Estimated optimal radius ~ 0.0366, so diameter ~ 0.0732
    # Hex spacing d ~ 0.0732
    # Try a few spacings around this
    best_points = None
    best_score = -1
    
    # Spacing range
    spacings = np.linspace(0.070, 0.078, 10)
    
    for d in spacings:
        # Hex grid parameters
        row_height = d * np.sqrt(3) / 2
        col_width = d
        
        # Estimate grid size
        # Square side 1.0
        # Number of rows ~ 1.0 / row_height
        # Number of cols ~ 1.0 / col_width
        
        # Generate points in a slightly larger box to ensure coverage
        # Centered at 0
        max_y = 0.5 + row_height
        max_x = 0.5 + col_width
        
        y_vals = np.arange(-max_y, max_y + row_height, row_height)
        points = []
        
        for j, y in enumerate(y_vals):
            if y < -0.5 - d or y > 0.5 + d:
                continue
            # Offset for odd rows
            offset = 0.0
            if j % 2 != 0:
                offset = col_width / 2.0
            
            x_vals = np.arange(-max_x + offset, max_x + offset + col_width, col_width)
            for x in x_vals:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        
        # Select n points.
        # Heuristic: Points closer to the boundary limit the radius more easily 
        # if the packing isn't perfect. 
        # However, simply taking the closest to center might leave holes.
        # A better heuristic for "max min dist": 
        # 1. Calculate distance to boundary for each point.
        # 2. Sort by distance to boundary (descending).
        # 3. Take the top n.
        # This ensures the selected points have the most "room" to move.
        
        dists_to_boundary = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        
        # Sort indices by distance to boundary, descending
        idx = np.argsort(dists_to_boundary)[::-1]
        selected = points[idx[:n]]
        
        # Calculate score
        diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(selected[:, 0]), 0.5 - np.abs(selected[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        score = min(min_pair / 2, min_boundary)
        
        if score > best_score:
            best_score = score
            best_points = selected.copy()
            
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    return best_points

def construct():
    # 1. Warm start
    points = _generate_hex_grid(N)
    
    # 2. Optimize using Projected Gradient Descent on a penalty function
    # Objective: Maximize r such that:
    # d_ij >= 2r for all i<j
    # b_i >= r for all i
    
    # We iterate on r. For a fixed r, we minimize the violation:
    # V = sum_{i<j, d_ij < 2r} (2r - d_ij)^2 + sum_{i, b_i < r} (r - b_i)^2
    
    # Initial radius estimate
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair / 2, min_boundary)
    
    r = calc_radius(points)
    
    # Optimization parameters
    max_iter = 3000
    # Learning rate schedule: start high, decay
    lr_start = 0.005
    lr_end = 0.0001
    r_increment = 0.0002
    
    for it in range(max_iter):
        # Decay learning rate
        lr = lr_start * (lr_end / lr_start) ** (it / max_iter)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Gradient of V wrt points
        # V_pair = sum_{i<j} max(0, 2r - d_ij)^2
        # dV/dp_i = sum_{j != i} 2 * max(0, 2r - d_ij) * (2r - d_ij) * d(2r - d_ij)/dp_i
        # d(2r - d_ij)/dp_i = - (p_i - p_j) / d_ij
        # So term is: -2 * max(0, 2r - d_ij) * (2r - d_ij) * (p_i - p_j) / d_ij
        # Let w_ij = 2 * max(0, 2r - d_ij) * (2r - d_ij) / d_ij
        # Then grad_i = - sum_{j != i} w_ij * (p_i - p_j)
        # Note: w_ij is symmetric.
        
        mask = dists < 2 * r
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        
        # Calculate weights
        # violation = 2r - dists
        violation = np.where(mask, 2 * r - dists, 0)
        # w_ij = 2 * violation * violation / dists
        weights = np.zeros_like(dists)
        weights[mask] = 2 * (violation[mask] ** 2) / safe_dists[mask]
        
        # Gradient from pairs:
        # grad_i = - sum_j w_ij * (p_i - p_j)
        # grad_i = - (sum_j w_ij) * p_i + sum_j w_ij * p_j
        # grad_i = - row_sum_i * p_i + (weights @ points)_i
        
        row_sums = np.sum(weights, axis=1)
        weighted_points = weights @ points
        pair_grad = -row_sums[:, np.newaxis] * points + weighted_points
        
        # Gradient from boundary
        # V_bound = sum_i max(0, r - b_i)^2
        # b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # This is non-differentiable at min switch and abs.
        # Approximation:
        # bx = 0.5 - |x|, by = 0.5 - |y|
        # viol_x = max(0, r - bx), viol_y = max(0, r - by)
        # If bx < by, b_i = bx. If by < bx, b_i = by.
        # For simplicity and robustness, we apply gradient from both if violated, 
        # but weighted by which one is smaller? 
        # Actually, the gradient of min(a,b) is subgradient. 
        # Let's just use the gradient of the active constraint.
        
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        # Active constraint is min(bx, by)
        # If bx <= by, active is bx. Grad wrt x: -sign(x) (if x!=0)
        # If by < bx, active is by. Grad wrt y: -sign(y)
        
        bound_grad = np.zeros_like(points)
        
        # Mask for x-active
        mask_x = bx <= by
        mask_y = ~mask_x
        
        viol_x = np.where(mask_x, np.maximum(0, r - bx), 0)
        viol_y = np.where(mask_y, np.maximum(0, r - by), 0)
        
        # dV/dx = 2 * viol * d(r-bx)/dx = 2 * viol * (-d bx/dx)
        # d bx/dx = -sign(x)
        # So dV/dx = 2 * viol * sign(x)
        # Wait, we want to MINIMIZE V. 
        # V = (r-bx)^2. dV/dx = 2(r-bx)(-dbx/dx) = 2(r-bx)(sign(x)).
        # Update: p_new = p_old - lr * grad.
        # If x > 0 and r > bx (too close to right wall), bx = 0.5-x.
        # r > 0.5-x => x > 0.5-r.
        # viol = r - (0.5-x) = x - (0.5-r).
        # grad_x = 2 * viol * 1 = 2 * viol.
        # p_x -= lr * 2 * viol. x decreases. Correct.
        
        grad_x = 2 * viol_x * np.sign(points[:, 0])
        grad_y = 2 * viol_y * np.sign(points[:, 1])
        
        bound_grad[:, 0] = grad_x
        bound_grad[:, 1] = grad_y
        
        total_grad = pair_grad + bound_grad
        
        # Update
        points -= lr * total_grad
        
        # Project to square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Check if we can increase r
        current_r = calc_radius(points)
        
        if current_r > r + 1e-6:
            r = current_r
        elif current_r < r - 1e-4:
            # If we became infeasible, we might need to reduce r slightly
            # or the forces will push us back.
            # For now, just let it be, forces should fix it.
            pass
            
    # Final calculation of score
    return points
# EVOLVE-BLOCK-END
